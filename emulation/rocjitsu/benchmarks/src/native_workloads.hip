// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file native_workloads.hip
/// @brief Deterministic native HIP controls for the rocjitsu benchmark suite.

#include <hip/hip_runtime.h>

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

constexpr std::uint64_t kSeed = 0x524A4954ULL;
constexpr std::size_t kCopyBytes = 32ULL * 1024ULL * 1024ULL;
constexpr std::size_t kCopyElements = kCopyBytes / sizeof(float);
constexpr std::size_t kVectorElements = 1'048'583;
constexpr unsigned int kBlockSize = 256;
constexpr unsigned int kNoopBlocks = 1;
constexpr unsigned int kNoopThreads = 64;
constexpr unsigned int kCopyGridSize = 256;
constexpr std::size_t kVectorBytes = 16;
constexpr std::string_view kBuildTarget = ROCJITSU_BENCHMARK_BUILD_TARGET;

using Clock = std::chrono::steady_clock;

struct Options {
  std::string case_id;
  std::string expected_target;
  std::string output = "-";
  unsigned int warmups = 3;
  unsigned int samples = 21;
  bool describe_case = false;
};

struct Result {
  Options options;
  std::string reported_target;
  std::string device_name;
  int runtime_version = 0;
  int driver_version = 0;
  std::vector<std::uint64_t> synchronized_dispatch_ns;
};

[[noreturn]] void fail(std::string message) { throw std::runtime_error(std::move(message)); }

void check_hip(hipError_t status, std::string_view expression) {
  if (status == hipSuccess)
    return;
  std::ostringstream message;
  message << expression << " failed: " << hipGetErrorString(status) << " ("
          << static_cast<int>(status) << ')';
  fail(message.str());
}

#define HIP_CHECK(expression) check_hip((expression), #expression)

template <typename T> class DeviceBuffer {
public:
  explicit DeviceBuffer(std::size_t elements) : elements_(elements) {
    HIP_CHECK(hipMalloc(reinterpret_cast<void **>(&pointer_), elements * sizeof(T)));
  }

  ~DeviceBuffer() {
    if (pointer_ != nullptr)
      (void)hipFree(pointer_);
  }

  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;

  [[nodiscard]] T *get() const { return pointer_; }
  [[nodiscard]] std::size_t bytes() const { return elements_ * sizeof(T); }

private:
  T *pointer_ = nullptr;
  std::size_t elements_ = 0;
};

extern "C" __global__ void launch_noop_kernel() {}

extern "C" __global__ void copy_u128_grid_stride_kernel(const uint4 *__restrict__ source,
                                                        uint4 *__restrict__ destination,
                                                        std::uint64_t count) {
  std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::uint64_t stride = static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
  for (; index < count; index += stride)
    destination[index] = source[index];
}

extern "C" __global__ void vector_add_tail_kernel(const float *__restrict__ left,
                                                  const float *__restrict__ right,
                                                  float *__restrict__ output, std::uint64_t count) {
  const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count)
    output[index] = left[index] + right[index];
}

std::string usage(std::string_view executable) {
  std::ostringstream output;
  output << "Usage: " << executable << " --case CASE --expected-target GFX [options]\n\n"
         << "Cases:\n"
         << "  hip.launch_noop\n"
         << "  hip.copy_fp32_32m\n"
         << "  hip.vector_add_fp32_tail\n\n"
         << "Options:\n"
         << "  --warmups N          Untimed synchronized launches (default: 3)\n"
         << "  --samples N          Timed synchronized launches (default: 21)\n"
         << "  --output PATH        JSON output path; '-' writes stdout\n"
         << "  --describe-case CASE  Print device-free workload metadata as JSON\n"
         << "  --help               Show this help\n";
  return output.str();
}

template <typename T> T parse_unsigned(std::string_view value, std::string_view option) {
  T parsed = 0;
  const char *begin = value.data();
  const char *end = value.data() + value.size();
  const auto [position, error] = std::from_chars(begin, end, parsed);
  if (value.empty() || error != std::errc{} || position != end)
    fail("invalid value for " + std::string(option) + ": " + std::string(value));
  return parsed;
}

std::string_view require_value(int &index, int argc, char **argv, std::string_view option) {
  if (++index >= argc)
    fail("missing value for " + std::string(option));
  return argv[index];
}

Options parse_options(int argc, char **argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument = argv[index];
    if (argument == "--help") {
      std::cout << usage(argv[0]);
      std::exit(0);
    }
    if (argument == "--case") {
      if (!options.case_id.empty())
        fail("--case and --describe-case are mutually exclusive");
      options.case_id = require_value(index, argc, argv, argument);
    } else if (argument == "--describe-case") {
      if (!options.case_id.empty())
        fail("--case and --describe-case are mutually exclusive");
      options.case_id = require_value(index, argc, argv, argument);
      options.describe_case = true;
    } else if (argument == "--expected-target") {
      options.expected_target = require_value(index, argc, argv, argument);
    } else if (argument == "--warmups") {
      options.warmups =
          parse_unsigned<unsigned int>(require_value(index, argc, argv, argument), argument);
    } else if (argument == "--samples") {
      options.samples =
          parse_unsigned<unsigned int>(require_value(index, argc, argv, argument), argument);
    } else if (argument == "--output") {
      options.output = require_value(index, argc, argv, argument);
    } else {
      fail("unknown option: " + std::string(argument));
    }
  }

  const std::vector<std::string_view> cases = {"hip.launch_noop", "hip.copy_fp32_32m",
                                               "hip.vector_add_fp32_tail"};
  if (std::find(cases.begin(), cases.end(), options.case_id) == cases.end())
    fail("--case must name one of the documented native workloads");
  if (!options.describe_case && options.expected_target.empty())
    fail("--expected-target is required");
  if (options.samples == 0)
    fail("--samples must be positive");
  if (options.output.empty())
    fail("--output must not be empty");
  return options;
}

bool target_matches(std::string_view reported, std::string_view expected) {
  if (!reported.starts_with(expected))
    return false;
  return reported.size() == expected.size() || reported[expected.size()] == ':';
}

std::uint64_t splitmix64(std::uint64_t &state) {
  std::uint64_t value = (state += 0x9e3779b97f4a7c15ULL);
  value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31U);
}

void fill_bytes(std::vector<float> &values) {
  auto *bytes = reinterpret_cast<std::uint8_t *>(values.data());
  const std::size_t byte_count = values.size() * sizeof(float);
  std::uint64_t state = kSeed;
  for (std::size_t offset = 0; offset < byte_count;) {
    const std::uint64_t value = splitmix64(state);
    const std::size_t count = std::min(sizeof(value), byte_count - offset);
    std::copy_n(reinterpret_cast<const std::uint8_t *>(&value), count, bytes + offset);
    offset += count;
  }
}

void fill_floats(std::vector<float> &values, std::uint64_t salt) {
  std::uint64_t state = kSeed ^ salt;
  for (float &value : values) {
    const std::uint32_t bits = static_cast<std::uint32_t>(splitmix64(state) >> 40U);
    value = static_cast<float>(bits) / static_cast<float>(1U << 24U) - 0.5F;
  }
}

template <typename Launch>
std::vector<std::uint64_t> measure(const Options &options, Launch launch) {
  for (unsigned int iteration = 0; iteration < options.warmups; ++iteration) {
    launch();
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
  }

  std::vector<std::uint64_t> durations;
  durations.reserve(options.samples);
  for (unsigned int sample = 0; sample < options.samples; ++sample) {
    const auto begin = Clock::now();
    launch();
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    const auto end = Clock::now();
    durations.push_back(static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count()));
  }
  return durations;
}

void run_noop(Result &result) {
  result.synchronized_dispatch_ns = measure(result.options, []() {
    launch_noop_kernel<<<dim3(kNoopBlocks), dim3(kNoopThreads)>>>();
  });
}

void run_copy(Result &result) {
  static_assert(kCopyBytes % sizeof(uint4) == 0);
  static_assert(sizeof(uint4) == kVectorBytes);
  std::vector<float> source(kCopyElements);
  fill_bytes(source);

  DeviceBuffer<float> device_source(kCopyElements);
  DeviceBuffer<float> device_output(kCopyElements);
  HIP_CHECK(
      hipMemcpy(device_source.get(), source.data(), device_source.bytes(), hipMemcpyHostToDevice));

  const auto launch = [&]() {
    copy_u128_grid_stride_kernel<<<dim3(kCopyGridSize), dim3(kBlockSize)>>>(
        reinterpret_cast<const uint4 *>(device_source.get()),
        reinterpret_cast<uint4 *>(device_output.get()), kCopyBytes / sizeof(uint4));
  };
  result.synchronized_dispatch_ns = measure(result.options, launch);
}

void run_vector_add(Result &result) {
  std::vector<float> left(kVectorElements);
  std::vector<float> right(kVectorElements);
  fill_floats(left, 0xA5A5ULL);
  fill_floats(right, 0x5A5AULL);

  DeviceBuffer<float> device_left(kVectorElements);
  DeviceBuffer<float> device_right(kVectorElements);
  DeviceBuffer<float> device_output(kVectorElements);
  HIP_CHECK(hipMemcpy(device_left.get(), left.data(), device_left.bytes(), hipMemcpyHostToDevice));
  HIP_CHECK(
      hipMemcpy(device_right.get(), right.data(), device_right.bytes(), hipMemcpyHostToDevice));

  constexpr std::size_t grid_size = (kVectorElements + kBlockSize - 1) / kBlockSize;
  const auto launch = [&]() {
    vector_add_tail_kernel<<<dim3(grid_size), dim3(kBlockSize)>>>(
        device_left.get(), device_right.get(), device_output.get(), kVectorElements);
  };
  result.synchronized_dispatch_ns = measure(result.options, launch);
}

std::string json_escape(std::string_view value) {
  std::ostringstream output;
  for (const unsigned char character : value) {
    switch (character) {
    case '"':
      output << "\\\"";
      break;
    case '\\':
      output << "\\\\";
      break;
    case '\n':
      output << "\\n";
      break;
    case '\r':
      output << "\\r";
      break;
    case '\t':
      output << "\\t";
      break;
    default:
      if (character < 0x20)
        output << "\\u" << std::hex << std::setw(4) << std::setfill('0')
               << static_cast<unsigned int>(character) << std::dec;
      else
        output << character;
    }
  }
  return output.str();
}

void write_workload_json(std::ostream &output, std::string_view case_id) {
  if (case_id == "hip.launch_noop") {
    output << "{\"blocks\": " << kNoopBlocks
           << ", \"threads_per_block\": " << kNoopThreads << '}';
  } else if (case_id == "hip.copy_fp32_32m") {
    output << "{\"block_size\": " << kBlockSize << ", \"bytes\": " << kCopyBytes
           << ", \"dtype\": \"fp32\", \"elements\": " << kCopyElements
           << ", \"grid_size\": " << kCopyGridSize
           << ", \"vector_bytes\": " << kVectorBytes << '}';
  } else {
    output << "{\"block_size\": " << kBlockSize << ", \"dtype\": \"fp32\", \"elements\": "
           << kVectorElements << "}";
  }
}

void write_json(std::ostream &output, const Result &result) {
  const auto &samples = result.synchronized_dispatch_ns;
  output << "{\n"
         << "  \"schema\": \"rocjitsu.benchmark.workload.v1\",\n"
         << "  \"case_id\": \"" << json_escape(result.options.case_id) << "\",\n"
         << "  \"target\": {\"expected\": \"" << json_escape(result.options.expected_target)
         << "\", \"compiled\": \"" << kBuildTarget << "\", \"reported\": \""
         << json_escape(result.reported_target) << "\"},\n"
         << "  \"seed\": \"0x524a4954\",\n"
         << "  \"warmups\": " << result.options.warmups << ",\n"
         << "  \"samples\": " << result.options.samples << ",\n"
         << "  \"synchronized_dispatch_ns\": [";
  for (std::size_t index = 0; index < samples.size(); ++index) {
    if (index != 0)
      output << ", ";
    output << samples[index];
  }
  output << "],\n"
         << "  \"workload\": ";
  write_workload_json(output, result.options.case_id);
  output << ",\n"
         << "  \"provenance\": {\"adapter\": \"native_hip\", \"compiler\": \""
         << json_escape(__VERSION__) << "\", \"hip_runtime_version\": " << result.runtime_version
         << ", \"hip_driver_version\": " << result.driver_version << ", \"device_name\": \""
         << json_escape(result.device_name) << "\"}\n"
         << "}\n";
}

Result run(const Options &options) {
  if (!target_matches(kBuildTarget, options.expected_target))
    fail("binary target '" + std::string(kBuildTarget) + "' does not match expected target '" +
         options.expected_target + "'");

  HIP_CHECK(hipSetDevice(0));
  hipDeviceProp_t properties{};
  HIP_CHECK(hipGetDeviceProperties(&properties, 0));

  Result result;
  result.options = options;
  result.device_name = properties.name;
  result.reported_target = properties.gcnArchName;
  HIP_CHECK(hipRuntimeGetVersion(&result.runtime_version));
  HIP_CHECK(hipDriverGetVersion(&result.driver_version));
  if (!target_matches(result.reported_target, options.expected_target))
    fail("runtime reported target '" + result.reported_target + "', expected '" +
         options.expected_target + "'");

  if (options.case_id == "hip.launch_noop")
    run_noop(result);
  else if (options.case_id == "hip.copy_fp32_32m")
    run_copy(result);
  else
    run_vector_add(result);

  (void)hipDeviceReset();
  return result;
}

} // namespace

int main(int argc, char **argv) {
  try {
    const Options options = parse_options(argc, argv);
    if (options.describe_case) {
      write_workload_json(std::cout, options.case_id);
      std::cout << '\n';
      return 0;
    }
    const Result result = run(options);
    if (options.output == "-") {
      write_json(std::cout, result);
    } else {
      std::ofstream output(options.output, std::ios::trunc);
      if (!output)
        fail("could not open output file: " + options.output);
      write_json(output, result);
    }
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "rocjitsu native benchmark: " << error.what() << '\n';
    return 2;
  }
}
