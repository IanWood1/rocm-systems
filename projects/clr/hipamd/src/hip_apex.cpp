/* Copyright (c) 2026 Advanced Micro Devices, Inc.
 *
 * APEX (Adaptive Prefetch EXtensions) - HIP Integration
 *
 * GPU memory tracking and proactive prefetch for MoE inference.
 * Zero-cost when APEX_GPU_ENABLE is not set.
 */

#include "hip_apex.h"

#include <hip/hip_runtime.h>
#include "hip_internal.hpp"
#include <atomic>
#include <cerrno>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <vector>
#include <cstdio>

namespace hip {
hipError_t ihipMalloc(void** ptr, size_t sizeBytes, unsigned int flags);
hipError_t ihipMallocManaged(void** ptr, size_t size, size_t align, bool use_host_ptr);
hipError_t ihipMemPrefetchAsync(const void* dev_ptr, size_t count, hipMemLocation location,
                                hipStream_t stream);
hipError_t ihipMemAdvise(const void* dev_ptr, size_t count, hipMemoryAdvise advice,
                         hipMemLocation location);
}  // namespace hip

namespace apex {

namespace {

constexpr size_t kMiB = 1024ull * 1024ull;
constexpr size_t kGiB = 1024ull * 1024ull * 1024ull;
constexpr size_t kRedirectMinSize = 256ull * kMiB;
constexpr uint32_t kMaxPrefetchDevices = 64;

struct HbmPrefetchConfig {
    uint64_t limit_bytes = 0;
    bool disable_all = false;
};

}  // namespace

// ============================================================
// Global state
// ============================================================

static std::atomic<int> g_enabled{-1};  // -1 = not checked, 0 = off, 1 = on
static std::atomic<int> g_debug{-1};
static std::atomic<int> g_managed_malloc_redirect{-1};
static std::atomic<uint64_t> g_managed_malloc_min_bytes{UINT64_MAX};
static std::atomic<int> g_prefetch_enabled{-1};
static std::atomic<bool> g_prefetch_logged{false};
static std::atomic<int> g_device_first_malloc{-1};

struct Allocation {
    void* ptr;
    size_t size;
    unsigned int flags;
    bool managed;
    int preferred_device;  // -1 = no preference
    int last_device;
    uint64_t access_count;
    size_t prefetched_bytes;
    int prefetch_device;
};

struct Expert {
    uint32_t id;
    void* ptr;
    size_t size;
    int current_device;     // -1 = system RAM, 0+ = GPU
    uint64_t last_access;   // monotonic counter
};

static std::mutex g_mutex;
static std::unordered_map<void*, Allocation> g_allocs;
static std::unordered_map<uint32_t, Expert> g_experts;
static Stats g_stats = {};
static uint64_t g_access_counter = 0;
static std::once_flag g_hbm_prefetch_once;
static HbmPrefetchConfig g_hbm_prefetch;
static std::atomic<uint64_t> g_prefetched_bytes_by_device[kMaxPrefetchDevices];
static std::atomic<uint64_t> g_prefetched_bytes_other{0};

static bool debug_enabled() {
    int d = g_debug.load(std::memory_order_relaxed);
    if (d < 0) {
        const char* env = getenv("APEX_GPU_DEBUG");
        d = (env && env[0] == '1') ? 1 : 0;
        g_debug.store(d, std::memory_order_relaxed);
    }
    return d == 1;
}

// ============================================================
// Public API
// ============================================================

bool enabled() {
    int e = g_enabled.load(std::memory_order_relaxed);
    if (e < 0) {
        const char* env = getenv("APEX_GPU_ENABLE");
        e = (env && env[0] == '1') ? 1 : 0;
        g_enabled.store(e, std::memory_order_relaxed);
        if (e) {
            fprintf(stderr, "[APEX] GPU integration enabled\n");
        }
    }
    return e == 1;
}

bool managed_malloc_redirect_enabled() {
    if (!enabled()) return false;

    int e = g_managed_malloc_redirect.load(std::memory_order_relaxed);
    if (e < 0) {
        const char* env = getenv("APEX_MANAGED_MALLOC");
        e = (env && env[0] == '1') ? 1 : 0;
        g_managed_malloc_redirect.store(e, std::memory_order_relaxed);
        if (e) {
            fprintf(stderr, "[APEX] hipMalloc -> hipMallocManaged redirect enabled\n");
        }
    }
    return e == 1;
}

static uint64_t parse_env_u64(const char* name, uint64_t default_value) {
    const char* env = getenv(name);
    if (!env || env[0] == '\0') {
        return default_value;
    }

    char* end = nullptr;
    unsigned long long parsed = strtoull(env, &end, 0);
    if (end == env || (end && end[0] != '\0')) {
        return default_value;
    }
    return static_cast<uint64_t>(parsed);
}

static uint64_t managed_malloc_min_bytes() {
    uint64_t cached = g_managed_malloc_min_bytes.load(std::memory_order_relaxed);
    if (cached != UINT64_MAX) {
        return cached;
    }

    uint64_t min_bytes = parse_env_u64("APEX_MANAGED_MALLOC_MIN_MB",
                                       kRedirectMinSize / kMiB) * kMiB;
    min_bytes = parse_env_u64("APEX_MANAGED_MALLOC_MIN_BYTES", min_bytes);
    g_managed_malloc_min_bytes.store(min_bytes, std::memory_order_relaxed);
    if (debug_enabled()) {
        fprintf(stderr, "[APEX] hipMalloc managed redirect min size=%llu bytes\n",
                static_cast<unsigned long long>(min_bytes));
    }
    return min_bytes;
}

bool should_redirect_malloc(size_t size) {
    return managed_malloc_redirect_enabled() && size >= managed_malloc_min_bytes();
}

static bool prefetch_enabled() {
    if (!enabled()) return false;

    int e = g_prefetch_enabled.load(std::memory_order_relaxed);
    if (e < 0) {
        const char* env = getenv("APEX_EXPERT_PREFETCH");
        e = (env && env[0] == '1') ? 1 : 0;
        g_prefetch_enabled.store(e, std::memory_order_relaxed);
    }
    if (e == 1 && !g_prefetch_logged.exchange(true, std::memory_order_relaxed)) {
        fprintf(stderr, "[APEX] expert prefetch tracking enabled\n");
    }
    return e == 1;
}

static bool device_first_malloc_enabled() {
    if (!enabled()) return false;

    int e = g_device_first_malloc.load(std::memory_order_relaxed);
    if (e < 0) {
        const char* policy = getenv("APEX_ALLOC_POLICY");
        const char* legacy = getenv("APEX_DEVICE_FIRST_MALLOC");
        e = ((policy && (strcmp(policy, "device_first") == 0 ||
                         strcmp(policy, "device_first_managed_fallback") == 0)) ||
             (legacy && legacy[0] == '1')) ? 1 : 0;
        g_device_first_malloc.store(e, std::memory_order_relaxed);
        if (e) {
            fprintf(stderr, "[APEX] allocation policy: device-first managed fallback\n");
        }
    }
    return e == 1;
}

static bool parse_hbm_limit_gb(const char* env, uint64_t* limit_bytes) {
    if (!env || env[0] == '\0' || env[0] == '-') return false;

    errno = 0;
    char* end = nullptr;
    unsigned long long parsed = strtoull(env, &end, 10);
    if (errno == ERANGE || end == env || !end || end[0] != '\0') return false;
    if (parsed > ULLONG_MAX / static_cast<unsigned long long>(kGiB)) return false;

    *limit_bytes = static_cast<uint64_t>(parsed) * kGiB;
    return true;
}

static const HbmPrefetchConfig& hbm_prefetch_config() {
    std::call_once(g_hbm_prefetch_once, []() {
        const char* env = getenv("APEX_HBM_LIMIT_GB");
        if (env && env[0] != '\0') {
            uint64_t limit_bytes = 0;
            if (!parse_hbm_limit_gb(env, &limit_bytes)) {
                fprintf(stderr, "[APEX] invalid APEX_HBM_LIMIT_GB=%s; using unlimited prefetch\n", env);
            } else if (limit_bytes == 0) {
                g_hbm_prefetch.disable_all = true;
                g_hbm_prefetch.limit_bytes = 0;
                fprintf(stderr, "[APEX] HBM_LIMIT_GB=0: all redirect prefetch disabled\n");
            } else {
                g_hbm_prefetch.limit_bytes = limit_bytes;
            }
        }
    });
    return g_hbm_prefetch;
}

static std::atomic<uint64_t>* prefetched_bytes_for_device(int device) {
    if (device >= 0 && device < static_cast<int>(kMaxPrefetchDevices)) {
        return &g_prefetched_bytes_by_device[device];
    }
    return &g_prefetched_bytes_other;
}

static void release_prefetch_budget(int device, size_t bytes) {
    if (!bytes) return;
    prefetched_bytes_for_device(device)->fetch_sub(bytes, std::memory_order_relaxed);
}

static size_t reserve_prefetch_budget(int device, size_t size, uint64_t limit_bytes,
                                      uint64_t* previous_total) {
    std::atomic<uint64_t>* counter = prefetched_bytes_for_device(device);
    if (previous_total) *previous_total = 0;

    if (limit_bytes == 0) {
        uint64_t previous = counter->fetch_add(size, std::memory_order_relaxed);
        if (previous_total) *previous_total = previous;
        return size;
    }

    for (;;) {
        uint64_t current = counter->load(std::memory_order_relaxed);
        if (current >= limit_bytes) {
            if (previous_total) *previous_total = current;
            return 0;
        }

        uint64_t remaining = limit_bytes - current;
        uint64_t reserved = remaining < static_cast<uint64_t>(size) ? remaining : size;
        if (counter->compare_exchange_weak(current, current + reserved,
                                           std::memory_order_relaxed)) {
            if (previous_total) *previous_total = current;
            return static_cast<size_t>(reserved);
        }
    }
}

static void maybe_apply_memory_hints(void* ptr, size_t size, int device, uint64_t hbm_limit) {
    hipMemLocation location = {};
    location.type = hipMemLocationTypeDevice;
    location.id = device;

    if (hbm_limit == 0) {
        (void)hip::ihipMemAdvise(ptr, size, hipMemAdviseSetPreferredLocation, location);
    }
    (void)hip::ihipMemAdvise(ptr, size, hipMemAdviseSetReadMostly, location);
    (void)hip::ihipMemAdvise(ptr, size, hipMemAdviseSetAccessedBy, location);
}

struct PrefetchResult {
    bool attempted = false;
    int device = -1;
    size_t prefetched_bytes = 0;
    hipError_t status = hipErrorInvalidValue;
};

struct DeviceFirstMallocResult {
    hipError_t status = hipErrorNotSupported;
    bool suppress_managed_prefetch = false;
};

static PrefetchResult maybe_prefetch_redirected_alloc(void* ptr, size_t size) {
    PrefetchResult result;
    int device = hip::ihipGetDevice();
    if (device < 0) device = 0;
    result.device = device;

    const HbmPrefetchConfig& config = hbm_prefetch_config();
    if (config.disable_all) {
        if (debug_enabled()) {
            fprintf(stderr, "[APEX] lazy redirect prefetch skip %zuMB\n", size / kMiB);
        }
        return result;
    }

    uint64_t previous_total = 0;
    size_t prefetch_size = reserve_prefetch_budget(device, size, config.limit_bytes, &previous_total);
    if (prefetch_size == 0) {
        if (debug_enabled()) {
            fprintf(stderr, "[APEX] redirect prefetch budget exhausted size=%zuMB device=%d\n",
                    size / kMiB, device);
        }
        return result;
    }

    maybe_apply_memory_hints(ptr, prefetch_size, device, config.limit_bytes);

    hipMemLocation location = {};
    location.type = hipMemLocationTypeDevice;
    location.id = device;
    result.attempted = true;
    result.status = hip::ihipMemPrefetchAsync(ptr, prefetch_size, location, nullptr);
    if (result.status == hipSuccess) {
        result.prefetched_bytes = prefetch_size;
    } else {
        release_prefetch_budget(device, prefetch_size);
    }

    if (debug_enabled()) {
        fprintf(stderr, "[APEX] redirect prefetch ptr=%p size=%zuMB/%zuMB device=%d previous=%lluMB status=%d\n",
                ptr, prefetch_size / kMiB, size / kMiB, device,
                static_cast<unsigned long long>(previous_total / kMiB), result.status);
    }
    return result;
}

static Allocation* find_alloc_unlocked(const void* ptr) {
    const char* candidate = static_cast<const char*>(ptr);
    for (auto& entry : g_allocs) {
        const Allocation& alloc = entry.second;
        const char* begin = static_cast<const char*>(alloc.ptr);
        const char* end = begin + alloc.size;
        if (candidate >= begin && candidate < end) {
            return &entry.second;
        }
    }
    return nullptr;
}

static void update_alloc_prefetch(void* ptr, int device, size_t prefetched) {
    if (!ptr || !prefetched) return;

    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_allocs.find(ptr);
    if (it != g_allocs.end()) {
        it->second.prefetch_device = device;
        it->second.prefetched_bytes += prefetched;
    }
}

static DeviceFirstMallocResult try_device_first_malloc(void** ptr, size_t size) {
    DeviceFirstMallocResult result;
    if (!device_first_malloc_enabled()) {
        return result;
    }

    int device = hip::ihipGetDevice();
    if (device < 0) device = 0;

    const HbmPrefetchConfig& config = hbm_prefetch_config();
    if (config.disable_all) {
        if (debug_enabled()) {
            fprintf(stderr, "[APEX] device-first malloc disabled by APEX_HBM_LIMIT_GB=0 size=%lluMB\n",
                    static_cast<unsigned long long>(size / kMiB));
        }
        return result;
    }

    uint64_t previous_total = 0;
    size_t reserved = reserve_prefetch_budget(device, size, config.limit_bytes, &previous_total);
    if (reserved < size) {
        if (reserved) {
            release_prefetch_budget(device, reserved);
        }
        if (debug_enabled()) {
            fprintf(stderr, "[APEX] device-first budget skip size=%zuMB reserved=%zuMB device=%d previous=%lluMB\n",
                    size / kMiB, reserved / kMiB, device,
                    static_cast<unsigned long long>(previous_total / kMiB));
        }
        return result;
    }

    hipError_t status = hip::ihipMalloc(ptr, size, 0);
    if (status != hipSuccess || ptr == nullptr || *ptr == nullptr) {
        release_prefetch_budget(device, reserved);
        if (ptr != nullptr) {
            *ptr = nullptr;
        }
        if (debug_enabled()) {
            fprintf(stderr, "[APEX] device-first malloc failed size=%zuMB device=%d previous=%lluMB status=%d\n",
                    size / kMiB, device,
                    static_cast<unsigned long long>(previous_total / kMiB), status);
        }
        result.status = status;
        result.suppress_managed_prefetch = true;
        return result;
    }

    update_alloc_prefetch(*ptr, device, reserved);
    if (debug_enabled()) {
        fprintf(stderr, "[APEX] hipMalloc(%zuMB) -> device-first %p device=%d previous=%lluMB\n",
                size / kMiB, *ptr, device,
                static_cast<unsigned long long>(previous_total / kMiB));
    }
    result.status = hipSuccess;
    return result;
}

hipError_t try_redirected_managed_malloc(void** ptr, size_t size) {
    if (!should_redirect_malloc(size)) {
        return hipErrorNotSupported;
    }

    DeviceFirstMallocResult device_result = try_device_first_malloc(ptr, size);
    if (device_result.status == hipSuccess) {
        return device_result.status;
    }

    hipError_t status = hip::ihipMallocManaged(ptr, size, 0, false);
    if (status != hipSuccess || ptr == nullptr || *ptr == nullptr) {
        if (debug_enabled()) {
            fprintf(stderr, "[APEX] hipMallocManaged redirect failed ptr=%p size=%zuMB status=%d\n",
                    ptr ? *ptr : nullptr, size / kMiB, status);
        }
        return status;
    }

    if (prefetch_enabled() && !device_result.suppress_managed_prefetch) {
        PrefetchResult prefetch = maybe_prefetch_redirected_alloc(*ptr, size);
        if (prefetch.prefetched_bytes) {
            update_alloc_prefetch(*ptr, prefetch.device, prefetch.prefetched_bytes);
        }
    } else if (device_result.suppress_managed_prefetch && debug_enabled()) {
        fprintf(stderr,
                "[APEX] managed redirect suppress allocation prefetch after device-first failure ptr=%p size=%zuMB\n",
                *ptr, size / kMiB);
    }

    if (debug_enabled()) {
        fprintf(stderr, "[APEX] hipMalloc(%zuMB) -> managed %p status=%d\n",
                size / kMiB, *ptr, status);
    }
    return status;
}

void track_alloc(void* ptr, size_t size, unsigned int flags, bool managed) {
    if (!ptr || !size) return;

    std::lock_guard<std::mutex> lock(g_mutex);
    g_allocs[ptr] = Allocation{ptr, size, flags, managed, -1, -1, 0, 0, -1};
    g_stats.allocs++;
    if (managed) {
        g_stats.managed_bytes += size;
    }

    if (debug_enabled()) {
        fprintf(stderr, "[APEX] alloc %p size=%zu managed=%d flags=0x%x\n",
                ptr, size, managed, flags);
    }
}

void track_free(void* ptr) {
    if (!ptr) return;

    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_allocs.find(ptr);
    if (it != g_allocs.end()) {
        if (it->second.prefetched_bytes) {
            release_prefetch_budget(it->second.prefetch_device, it->second.prefetched_bytes);
        }
        if (it->second.managed) {
            g_stats.managed_bytes -= it->second.size;
        }
        g_allocs.erase(it);
        g_stats.frees++;

        if (debug_enabled()) {
            fprintf(stderr, "[APEX] free %p\n", ptr);
        }
    }
}

void pre_launch(const void* host_function, void** args) {
    // In a full implementation, this would:
    // 1. Look up the kernel by host_function pointer
    // 2. Inspect kernel arguments to find pointers to tracked allocations
    // 3. For MoE expert kernels, prefetch selected expert weights to HBM
    //
    // For now, we just count launches for statistics
    (void)host_function;
    (void)args;

    if (debug_enabled()) {
        fprintf(stderr, "[APEX] pre_launch func=%p\n", host_function);
    }
}

void record_prefetch(const void* ptr, size_t count, int device) {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_stats.prefetches_triggered++;

    Allocation* alloc = find_alloc_unlocked(ptr);
    if (alloc) {
        alloc->last_device = device;
        alloc->access_count++;
    }

    if (debug_enabled()) {
        fprintf(stderr, "[APEX] prefetch %p size=%zu device=%d\n", ptr, count, device);
    }
}

void register_expert(uint32_t expert_id, void* ptr, size_t size) {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_experts[expert_id] = Expert{expert_id, ptr, size, -1, 0};
    g_stats.experts_registered++;

    if (debug_enabled()) {
        fprintf(stderr, "[APEX] register expert %u ptr=%p size=%zu\n",
                expert_id, ptr, size);
    }
}

void select_experts(const uint32_t* expert_ids, uint32_t count) {
    std::unique_lock<std::mutex> lock(g_mutex);
    uint64_t access = ++g_access_counter;

    for (uint32_t i = 0; i < count; i++) {
        auto it = g_experts.find(expert_ids[i]);
        if (it == g_experts.end()) continue;

        Expert& exp = it->second;
        exp.last_access = access;

        if (exp.current_device >= 0) {
            // Expert already in HBM - cache hit
            g_stats.expert_cache_hits++;
            continue;
        }

        // Expert in system RAM - need to prefetch. Release the mutex while
        // issuing the prefetch because ihipMemPrefetchAsync records APEX
        // feedback through this same mutex.
        g_stats.expert_cache_misses++;
        uint32_t id = exp.id;
        void* ptr = exp.ptr;
        size_t size = exp.size;
        lock.unlock();

        hipMemLocation location = {};
        location.type = hipMemLocationTypeDevice;
        location.id = 0;
        hipError_t err = hip::ihipMemPrefetchAsync(ptr, size, location, nullptr);

        lock.lock();
        auto update = g_experts.find(id);
        if (err == hipSuccess) {
            if (update != g_experts.end()) {
                update->second.current_device = 0;
            }
            g_stats.prefetches_triggered++;
            if (debug_enabled()) {
                fprintf(stderr, "[APEX] prefetch expert %u (%zu bytes) to GPU 0\n", id, size);
            }
        } else if (debug_enabled()) {
            fprintf(stderr, "[APEX] prefetch expert %u FAILED: %s\n", id, hipGetErrorString(err));
        }
    }
}

Stats get_stats() {
    std::lock_guard<std::mutex> lock(g_mutex);
    return g_stats;
}

}  // namespace apex
