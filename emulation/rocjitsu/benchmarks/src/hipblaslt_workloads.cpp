// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file hipblaslt_workloads.cpp
/// @brief Fixed hipBLASLt workloads for synchronized rocjitsu measurements.

#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>

#include <dlfcn.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "json_output.h"

namespace {

constexpr std::size_t kMaximumWorkspaceBytes = 32ULL * 1024ULL * 1024ULL;
constexpr unsigned int kDefaultWarmups = 3;
constexpr unsigned int kDefaultSamples = 21;

using Clock = std::chrono::steady_clock;

struct CaseSpec {
  std::string_view id;
  std::int64_t m;
  std::int64_t n;
  std::int64_t k;
  hipblasOperation_t trans_a;
  hipblasOperation_t trans_b;
  std::int32_t batch_count;
  hipDataType input_type;
  hipDataType output_type;
  std::string_view input_dtype;
  std::string_view output_dtype;
  bool scalar_scales;
};

constexpr std::array<CaseSpec, 3> kCases = {{
    {
        .id = "tensile.gemm_fp16",
        .m = 512,
        .n = 512,
        .k = 512,
        .trans_a = HIPBLAS_OP_N,
        .trans_b = HIPBLAS_OP_N,
        .batch_count = 1,
        .input_type = HIP_R_16F,
        .output_type = HIP_R_16F,
        .input_dtype = "fp16",
        .output_dtype = "fp16",
        .scalar_scales = false,
    },
    {
        .id = "tensile.gemm_bf16_batched",
        .m = 256,
        .n = 77,
        .k = 160,
        .trans_a = HIPBLAS_OP_N,
        .trans_b = HIPBLAS_OP_T,
        .batch_count = 16,
        .input_type = HIP_R_16BF,
        .output_type = HIP_R_16BF,
        .input_dtype = "bf16",
        .output_dtype = "bf16",
        .scalar_scales = false,
    },
    {
        .id = "tensile.gemm_fp8_scaled",
        .m = 128,
        .n = 128,
        .k = 128,
        .trans_a = HIPBLAS_OP_T,
        .trans_b = HIPBLAS_OP_N,
        .batch_count = 1,
        .input_type = HIP_R_8F_E4M3,
        .output_type = HIP_R_16BF,
        .input_dtype = "fp8",
        .output_dtype = "bf16",
        .scalar_scales = true,
    },
}};

struct Options {
  std::string case_id;
  std::string expected_target;
  std::string output = "-";
  unsigned int warmups = kDefaultWarmups;
  unsigned int samples = kDefaultSamples;
  bool describe_case = false;
  bool probe = false;
};

struct ProbeResult {
  std::string expected_target;
  std::string reported_target;
  std::string device_name;
  std::string hipblaslt_library_path;
  int hipblaslt_version = 0;
  int hip_runtime_version = 0;
  int hip_driver_version = 0;
};

struct Result {
  Options options;
  const CaseSpec *problem = nullptr;
  std::string reported_target;
  std::string device_name;
  std::string hipblaslt_library_path;
  int hipblaslt_version = 0;
  int hip_runtime_version = 0;
  int hip_driver_version = 0;
  std::size_t workspace_bytes = 0;
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

void check_hipblaslt(hipblasStatus_t status, std::string_view expression) {
  if (status == HIPBLAS_STATUS_SUCCESS)
    return;
  std::ostringstream message;
  message << expression << " failed with hipBLASLt status " << static_cast<int>(status);
  fail(message.str());
}

#define HIP_CHECK(expression) check_hip((expression), #expression)
#define HIPBLASLT_CHECK(expression) check_hipblaslt((expression), #expression)

std::string usage(std::string_view executable) {
  std::ostringstream output;
  output << "Usage:\n"
         << "  " << executable
         << " --case CASE --expected-target GFX [options]\n"
         << "  " << executable << " --probe --expected-target GFX [--output PATH]\n"
         << "  " << executable << " --describe-case CASE\n\n"
         << "Cases:\n"
         << "  tensile.gemm_fp16\n"
         << "  tensile.gemm_bf16_batched\n"
         << "  tensile.gemm_fp8_scaled\n\n"
         << "Options:\n"
         << "  --expected-target GFX Required runtime architecture\n"
         << "  --warmups N           Untimed synchronized launches (default: 3)\n"
         << "  --samples N           Timed synchronized launches (default: 21)\n"
         << "  --output PATH         JSON output path; '-' writes stdout\n"
         << "  --describe-case CASE Print device-free workload metadata as JSON\n"
         << "  --probe               Report target and hipBLASLt library identity\n"
         << "  --help                Show this help\n";
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

const CaseSpec *find_case(std::string_view case_id) {
  const auto found = std::find_if(kCases.begin(), kCases.end(), [case_id](const CaseSpec &problem) {
    return problem.id == case_id;
  });
  return found == kCases.end() ? nullptr : &*found;
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
      if (!options.case_id.empty() || options.probe)
        fail("--case, --describe-case, and --probe are mutually exclusive");
      options.case_id = require_value(index, argc, argv, argument);
    } else if (argument == "--describe-case") {
      if (!options.case_id.empty() || options.probe)
        fail("--case, --describe-case, and --probe are mutually exclusive");
      options.case_id = require_value(index, argc, argv, argument);
      options.describe_case = true;
    } else if (argument == "--probe") {
      if (!options.case_id.empty() || options.probe)
        fail("--case, --describe-case, and --probe are mutually exclusive");
      options.probe = true;
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

  if (!options.probe && find_case(options.case_id) == nullptr)
    fail("--case must name one of the documented hipBLASLt workloads");
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

std::string_view operation_name(hipblasOperation_t operation) {
  return operation == HIPBLAS_OP_N ? "N" : "T";
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
        output << "\\u" << std::hex << std::setfill('0') << std::setw(4)
               << static_cast<unsigned int>(character) << std::dec;
      else
        output << character;
    }
  }
  return output.str();
}

void write_workload_json(std::ostream &output, const CaseSpec &problem) {
  output << "{\"m\": " << problem.m << ", \"n\": " << problem.n << ", \"k\": "
         << problem.k << ", \"trans_a\": \"" << operation_name(problem.trans_a)
         << "\", \"trans_b\": \"" << operation_name(problem.trans_b)
         << "\", \"batch_count\": " << problem.batch_count << ", \"input_dtype\": \""
         << problem.input_dtype << "\", \"output_dtype\": \"" << problem.output_dtype
         << "\", \"accumulator_dtype\": \"fp32\"";
  if (problem.scalar_scales)
    output << ", \"scalar_scales\": true";
  output << ", \"beta\": 0}";
}

std::size_t checked_product(std::size_t left, std::size_t right, std::string_view name) {
  if (right != 0 && left > std::numeric_limits<std::size_t>::max() / right)
    fail(std::string(name) + " size overflows size_t");
  return left * right;
}

std::size_t element_size(hipDataType type) {
  switch (type) {
  case HIP_R_16F:
  case HIP_R_16BF:
    return 2;
  case HIP_R_8F_E4M3:
    return 1;
  default:
    fail("unsupported workload data type");
  }
}

class DeviceAllocation {
public:
  DeviceAllocation() = default;

  ~DeviceAllocation() {
    if (pointer_ != nullptr)
      (void)hipFree(pointer_);
  }

  DeviceAllocation(const DeviceAllocation &) = delete;
  DeviceAllocation &operator=(const DeviceAllocation &) = delete;

  void allocate(std::size_t bytes) {
    if (pointer_ != nullptr)
      fail("device allocation was initialized more than once");
    bytes_ = bytes;
    if (bytes != 0)
      HIP_CHECK(hipMalloc(&pointer_, bytes));
  }

  void fill_matrix(hipDataType type, std::size_t elements) {
    const std::size_t expected_bytes = checked_product(elements, element_size(type), "matrix");
    if (expected_bytes != bytes_)
      fail("matrix initialization size does not match its allocation");
    std::vector<std::uint8_t> host(bytes_);
    if (type == HIP_R_16F) {
      for (std::size_t offset = 0; offset < bytes_; offset += 2) {
        host[offset] = 0x00;
        host[offset + 1] = 0x3c; // IEEE binary16 1.0.
      }
    } else if (type == HIP_R_16BF) {
      for (std::size_t offset = 0; offset < bytes_; offset += 2) {
        host[offset] = 0x80;
        host[offset + 1] = 0x3f; // bfloat16 1.0.
      }
    } else {
      std::fill(host.begin(), host.end(), 0x38); // E4M3 1.0.
    }
    HIP_CHECK(hipMemcpy(pointer_, host.data(), bytes_, hipMemcpyHostToDevice));
  }

  template <typename T> void copy_scalar(const T &value) {
    if (bytes_ != sizeof(T))
      fail("scalar initialization size does not match its allocation");
    HIP_CHECK(hipMemcpy(pointer_, &value, sizeof(T), hipMemcpyHostToDevice));
  }

  [[nodiscard]] void *get() const { return pointer_; }
  [[nodiscard]] std::size_t bytes() const { return bytes_; }

private:
  void *pointer_ = nullptr;
  std::size_t bytes_ = 0;
};

class HipblasLtHandle {
public:
  HipblasLtHandle() { HIPBLASLT_CHECK(hipblasLtCreate(&handle_)); }

  ~HipblasLtHandle() {
    if (handle_ != nullptr)
      (void)hipblasLtDestroy(handle_);
  }

  HipblasLtHandle(const HipblasLtHandle &) = delete;
  HipblasLtHandle &operator=(const HipblasLtHandle &) = delete;

  [[nodiscard]] hipblasLtHandle_t get() const { return handle_; }

private:
  hipblasLtHandle_t handle_ = nullptr;
};

class HipStream {
public:
  HipStream() { HIP_CHECK(hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking)); }

  ~HipStream() {
    if (stream_ != nullptr)
      (void)hipStreamDestroy(stream_);
  }

  HipStream(const HipStream &) = delete;
  HipStream &operator=(const HipStream &) = delete;

  [[nodiscard]] hipStream_t get() const { return stream_; }

private:
  hipStream_t stream_ = nullptr;
};

class MatrixLayout {
public:
  MatrixLayout() = default;

  ~MatrixLayout() {
    if (layout_ != nullptr)
      (void)hipblasLtMatrixLayoutDestroy(layout_);
  }

  MatrixLayout(const MatrixLayout &) = delete;
  MatrixLayout &operator=(const MatrixLayout &) = delete;

  void create(hipDataType type, std::int64_t rows, std::int64_t columns) {
    HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(
        &layout_, type, static_cast<std::uint64_t>(rows), static_cast<std::uint64_t>(columns), rows));
  }

  void set_batch(std::int32_t count, std::int64_t stride) {
    if (count <= 1)
      return;
    HIPBLASLT_CHECK(hipblasLtMatrixLayoutSetAttribute(
        layout_, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &count, sizeof(count)));
    HIPBLASLT_CHECK(hipblasLtMatrixLayoutSetAttribute(
        layout_, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride, sizeof(stride)));
  }

  [[nodiscard]] hipblasLtMatrixLayout_t get() const { return layout_; }

private:
  hipblasLtMatrixLayout_t layout_ = nullptr;
};

class MatmulDescriptor {
public:
  MatmulDescriptor() = default;

  ~MatmulDescriptor() {
    if (descriptor_ != nullptr)
      (void)hipblasLtMatmulDescDestroy(descriptor_);
  }

  MatmulDescriptor(const MatmulDescriptor &) = delete;
  MatmulDescriptor &operator=(const MatmulDescriptor &) = delete;

  void create(hipblasOperation_t trans_a, hipblasOperation_t trans_b) {
    HIPBLASLT_CHECK(
        hipblasLtMatmulDescCreate(&descriptor_, HIPBLAS_COMPUTE_32F, HIP_R_32F));
    const std::int32_t trans_a_value = static_cast<std::int32_t>(trans_a);
    const std::int32_t trans_b_value = static_cast<std::int32_t>(trans_b);
    HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(
        descriptor_, HIPBLASLT_MATMUL_DESC_TRANSA, &trans_a_value, sizeof(trans_a_value)));
    HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(
        descriptor_, HIPBLASLT_MATMUL_DESC_TRANSB, &trans_b_value, sizeof(trans_b_value)));
  }

  void set_scalar_scales(void *scale_a, void *scale_b) {
    const auto mode = static_cast<std::uint32_t>(HIPBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F);
    HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(
        descriptor_, HIPBLASLT_MATMUL_DESC_A_SCALE_MODE, &mode, sizeof(mode)));
    HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(
        descriptor_, HIPBLASLT_MATMUL_DESC_B_SCALE_MODE, &mode, sizeof(mode)));
    HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(
        descriptor_, HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER, &scale_a, sizeof(scale_a)));
    HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(
        descriptor_, HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER, &scale_b, sizeof(scale_b)));
  }

  [[nodiscard]] hipblasLtMatmulDesc_t get() const { return descriptor_; }

private:
  hipblasLtMatmulDesc_t descriptor_ = nullptr;
};

class MatmulPreference {
public:
  MatmulPreference() { HIPBLASLT_CHECK(hipblasLtMatmulPreferenceCreate(&preference_)); }

  ~MatmulPreference() {
    if (preference_ != nullptr)
      (void)hipblasLtMatmulPreferenceDestroy(preference_);
  }

  MatmulPreference(const MatmulPreference &) = delete;
  MatmulPreference &operator=(const MatmulPreference &) = delete;

  void set_maximum_workspace(std::size_t bytes) {
    const std::uint64_t value = bytes;
    HIPBLASLT_CHECK(hipblasLtMatmulPreferenceSetAttribute(
        preference_, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &value, sizeof(value)));
  }

  [[nodiscard]] hipblasLtMatmulPreference_t get() const { return preference_; }

private:
  hipblasLtMatmulPreference_t preference_ = nullptr;
};

class MatmulProblem {
public:
  MatmulProblem(const CaseSpec &problem, hipblasLtHandle_t handle)
      : handle_(handle) {
    const std::int64_t a_rows = problem.trans_a == HIPBLAS_OP_N ? problem.m : problem.k;
    const std::int64_t a_columns = problem.trans_a == HIPBLAS_OP_N ? problem.k : problem.m;
    const std::int64_t b_rows = problem.trans_b == HIPBLAS_OP_N ? problem.k : problem.n;
    const std::int64_t b_columns = problem.trans_b == HIPBLAS_OP_N ? problem.n : problem.k;

    a_layout_.create(problem.input_type, a_rows, a_columns);
    b_layout_.create(problem.input_type, b_rows, b_columns);
    c_layout_.create(problem.output_type, problem.m, problem.n);
    d_layout_.create(problem.output_type, problem.m, problem.n);

    const std::int64_t a_stride = a_rows * a_columns;
    const std::int64_t b_stride = b_rows * b_columns;
    const std::int64_t output_stride = problem.m * problem.n;
    a_layout_.set_batch(problem.batch_count, a_stride);
    b_layout_.set_batch(problem.batch_count, b_stride);
    c_layout_.set_batch(problem.batch_count, output_stride);
    d_layout_.set_batch(problem.batch_count, output_stride);

    descriptor_.create(problem.trans_a, problem.trans_b);
    preference_.set_maximum_workspace(kMaximumWorkspaceBytes);

    const auto batches = static_cast<std::size_t>(problem.batch_count);
    const std::size_t a_elements = checked_product(static_cast<std::size_t>(a_stride), batches, "A");
    const std::size_t b_elements = checked_product(static_cast<std::size_t>(b_stride), batches, "B");
    const std::size_t output_elements =
        checked_product(static_cast<std::size_t>(output_stride), batches, "output");

    a_.allocate(checked_product(a_elements, element_size(problem.input_type), "A"));
    b_.allocate(checked_product(b_elements, element_size(problem.input_type), "B"));
    c_.allocate(checked_product(output_elements, element_size(problem.output_type), "C"));
    d_.allocate(checked_product(output_elements, element_size(problem.output_type), "D"));
    a_.fill_matrix(problem.input_type, a_elements);
    b_.fill_matrix(problem.input_type, b_elements);

    if (problem.scalar_scales) {
      constexpr float scale = 1.0F;
      scale_a_.allocate(sizeof(scale));
      scale_b_.allocate(sizeof(scale));
      scale_a_.copy_scalar(scale);
      scale_b_.copy_scalar(scale);
      descriptor_.set_scalar_scales(scale_a_.get(), scale_b_.get());
    }

    int returned_algorithms = 0;
    HIPBLASLT_CHECK(hipblasLtMatmulAlgoGetHeuristic(handle_,
                                                    descriptor_.get(),
                                                    a_layout_.get(),
                                                    b_layout_.get(),
                                                    c_layout_.get(),
                                                    d_layout_.get(),
                                                    preference_.get(),
                                                    1,
                                                    &heuristic_,
                                                    &returned_algorithms));
    if (returned_algorithms != 1)
      fail("hipBLASLt did not return a heuristic algorithm for " + std::string(problem.id));
    HIPBLASLT_CHECK(heuristic_.state);
    if (heuristic_.workspaceSize > kMaximumWorkspaceBytes)
      fail("hipBLASLt heuristic exceeded the configured workspace limit");
    workspace_.allocate(heuristic_.workspaceSize);
  }

  void dispatch_and_synchronize() {
    HIPBLASLT_CHECK(hipblasLtMatmul(handle_,
                                    descriptor_.get(),
                                    &alpha_,
                                    a_.get(),
                                    a_layout_.get(),
                                    b_.get(),
                                    b_layout_.get(),
                                    &beta_,
                                    c_.get(),
                                    c_layout_.get(),
                                    d_.get(),
                                    d_layout_.get(),
                                    &heuristic_.algo,
                                    workspace_.get(),
                                    workspace_.bytes(),
                                    stream_.get()));
    HIP_CHECK(hipStreamSynchronize(stream_.get()));
  }

  [[nodiscard]] std::size_t workspace_bytes() const { return workspace_.bytes(); }

private:
  hipblasLtHandle_t handle_ = nullptr;
  HipStream stream_;
  MatrixLayout a_layout_;
  MatrixLayout b_layout_;
  MatrixLayout c_layout_;
  MatrixLayout d_layout_;
  MatmulDescriptor descriptor_;
  MatmulPreference preference_;
  DeviceAllocation a_;
  DeviceAllocation b_;
  DeviceAllocation c_;
  DeviceAllocation d_;
  DeviceAllocation scale_a_;
  DeviceAllocation scale_b_;
  DeviceAllocation workspace_;
  hipblasLtMatmulHeuristicResult_t heuristic_{};
  float alpha_ = 1.0F;
  float beta_ = 0.0F;
};

std::string hipblaslt_library_path() {
  dlerror();
  void *symbol = dlsym(RTLD_DEFAULT, "hipblasLtCreate");
  const char *symbol_error = dlerror();
  if (symbol_error != nullptr || symbol == nullptr)
    fail("could not resolve hipblasLtCreate in the loaded hipBLASLt library");
  Dl_info information{};
  if (dladdr(symbol, &information) == 0 || information.dli_fname == nullptr)
    fail("could not resolve the loaded hipBLASLt library path");
  std::error_code error;
  const auto canonical = std::filesystem::weakly_canonical(information.dli_fname, error);
  return error ? std::string(information.dli_fname) : canonical.string();
}

ProbeResult probe(const Options &options) {
  HIP_CHECK(hipSetDevice(0));
  hipDeviceProp_t properties{};
  HIP_CHECK(hipGetDeviceProperties(&properties, 0));
  if (!target_matches(properties.gcnArchName, options.expected_target))
    fail("runtime reported target '" + std::string(properties.gcnArchName) + "', expected '" +
         options.expected_target + "'");

  HipblasLtHandle handle;
  ProbeResult result;
  result.expected_target = options.expected_target;
  result.reported_target = properties.gcnArchName;
  result.device_name = properties.name;
  result.hipblaslt_library_path = hipblaslt_library_path();
  HIPBLASLT_CHECK(hipblasLtGetVersion(handle.get(), &result.hipblaslt_version));
  HIP_CHECK(hipRuntimeGetVersion(&result.hip_runtime_version));
  HIP_CHECK(hipDriverGetVersion(&result.hip_driver_version));
  return result;
}

std::vector<std::uint64_t> measure(MatmulProblem &problem, const Options &options) {
  for (unsigned int iteration = 0; iteration < options.warmups; ++iteration)
    problem.dispatch_and_synchronize();

  std::vector<std::uint64_t> durations;
  durations.reserve(options.samples);
  for (unsigned int sample = 0; sample < options.samples; ++sample) {
    const auto begin = Clock::now();
    problem.dispatch_and_synchronize();
    const auto end = Clock::now();
    durations.push_back(static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count()));
  }
  return durations;
}

Result run(const Options &options) {
  const CaseSpec *problem = find_case(options.case_id);
  if (problem == nullptr)
    fail("unknown hipBLASLt workload");

  HIP_CHECK(hipSetDevice(0));
  hipDeviceProp_t properties{};
  HIP_CHECK(hipGetDeviceProperties(&properties, 0));
  if (!target_matches(properties.gcnArchName, options.expected_target))
    fail("runtime reported target '" + std::string(properties.gcnArchName) + "', expected '" +
         options.expected_target + "'");

  HipblasLtHandle handle;
  Result result;
  result.options = options;
  result.problem = problem;
  result.reported_target = properties.gcnArchName;
  result.device_name = properties.name;
  result.hipblaslt_library_path = hipblaslt_library_path();
  HIPBLASLT_CHECK(hipblasLtGetVersion(handle.get(), &result.hipblaslt_version));
  HIP_CHECK(hipRuntimeGetVersion(&result.hip_runtime_version));
  HIP_CHECK(hipDriverGetVersion(&result.hip_driver_version));

  MatmulProblem matmul(*problem, handle.get());
  result.workspace_bytes = matmul.workspace_bytes();
  result.synchronized_dispatch_ns = measure(matmul, options);
  return result;
}

void write_json(std::ostream &output, const Result &result) {
  const auto &samples = result.synchronized_dispatch_ns;
  output << "{\n"
         << "  \"schema\": \"rocjitsu.benchmark.workload.v1\",\n"
         << "  \"case_id\": \"" << json_escape(result.options.case_id) << "\",\n"
         << "  \"target\": {\"expected\": \"" << json_escape(result.options.expected_target)
         << "\", \"reported\": \"" << json_escape(result.reported_target) << "\"},\n"
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
  write_workload_json(output, *result.problem);
  output << ",\n"
         << "  \"provenance\": {\"adapter\": \"hipblaslt_cpp\", "
         << "\"algorithm_policy\": \"first_heuristic\", "
         << "\"deterministic_initialization\": \"fixed_one\", "
         << "\"hipblaslt_version\": " << result.hipblaslt_version
         << ", \"hipblaslt_library_path\": \""
         << json_escape(result.hipblaslt_library_path) << "\", \"workspace_bytes\": "
         << result.workspace_bytes
         << ", \"hip_runtime_version\": " << result.hip_runtime_version
         << ", \"hip_driver_version\": " << result.hip_driver_version
         << ", \"device_name\": \"" << json_escape(result.device_name) << "\"}\n"
         << "}\n";
}

void write_probe_json(std::ostream &output, const ProbeResult &result) {
  output << "{\n"
         << "  \"schema\": \"rocjitsu.benchmark.hipblaslt-probe.v1\",\n"
         << "  \"target\": {\"expected\": \"" << json_escape(result.expected_target)
         << "\", \"reported\": \"" << json_escape(result.reported_target) << "\"},\n"
         << "  \"hipblaslt_version\": " << result.hipblaslt_version << ",\n"
         << "  \"hipblaslt_library_path\": \""
         << json_escape(result.hipblaslt_library_path) << "\",\n"
         << "  \"hip_runtime_version\": " << result.hip_runtime_version << ",\n"
         << "  \"hip_driver_version\": " << result.hip_driver_version << ",\n"
         << "  \"device_name\": \"" << json_escape(result.device_name) << "\"\n"
         << "}\n";
}

template <typename Value, typename Writer>
void write_output(const std::string &path, const Value &value, Writer writer) {
  if (path == "-") {
    writer(std::cout, value);
    return;
  }
  std::ofstream output(path, std::ios::trunc);
  if (!output)
    fail("could not open output file: " + path);
  writer(output, value);
}

} // namespace

int main(int argc, char **argv) {
  try {
    const Options options = parse_options(argc, argv);
    if (options.probe) {
      write_output(options.output, probe(options), write_probe_json);
      return 0;
    }
    const CaseSpec *problem = find_case(options.case_id);
    if (options.describe_case) {
      write_workload_json(std::cout, *problem);
      std::cout << '\n';
      return 0;
    }

    const Result result = run(options);
    write_output(options.output, result, write_json);
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "rocjitsu hipBLASLt benchmark: " << error.what() << '\n';
    return 2;
  }
}
