# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# This directory is entered only when RJ_BUILD_BENCHMARKS is enabled by the
# parent project. Keep every heavyweight benchmark dependency behind that
# option so a normal rocjitsu build remains unchanged.

include(rj_find_amdcxx)
rj_find_amdcxx(RJ_BENCHMARK_AMDCXX)
if(NOT RJ_BENCHMARK_AMDCXX)
    message(
        FATAL_ERROR
        "RJ_BUILD_BENCHMARKS requires amdclang++ under ROCM_PATH='${ROCM_PATH}'."
    )
endif()

file(REAL_PATH "${ROCM_PATH}" _rj_benchmark_rocm_root)
function(_rj_require_benchmark_sdk_path value description)
    file(REAL_PATH "${value}" _resolved)
    cmake_path(
        IS_PREFIX _rj_benchmark_rocm_root "${_resolved}" NORMALIZE _inside_sdk
    )
    if(NOT _inside_sdk)
        message(
            FATAL_ERROR
            "${description} '${_resolved}' is outside ROCM_PATH "
            "'${_rj_benchmark_rocm_root}'."
        )
    endif()
endfunction()

foreach(_package IN ITEMS hip hipblas-common hipblaslt)
    # ROCM_PATH is the authoritative SDK for benchmark builds. Discard stale
    # package-cache entries so a reused build tree cannot mix ROCm installs.
    unset("${_package}_DIR")
    unset("${_package}_DIR" CACHE)
    find_package(
        ${_package}
        CONFIG
        REQUIRED
        PATHS "${ROCM_PATH}"
        NO_DEFAULT_PATH
    )
    set(_package_dir_variable "${_package}_DIR")
    _rj_require_benchmark_sdk_path(
        "${${_package_dir_variable}}" "${_package} package directory"
    )
endforeach()
unset(_package)
unset(_package_dir_variable)

get_target_property(
    _hipblaslt_location roc::hipblaslt IMPORTED_LOCATION_RELEASE
)
_rj_require_benchmark_sdk_path(
    "${_hipblaslt_location}" "hipBLASLt imported library"
)
get_target_property(
    _hipblaslt_includes roc::hipblaslt INTERFACE_INCLUDE_DIRECTORIES
)
foreach(_include IN LISTS _hipblaslt_includes)
    _rj_require_benchmark_sdk_path("${_include}" "hipBLASLt include directory")
endforeach()
unset(_hipblaslt_location)
unset(_hipblaslt_includes)
unset(_include)

add_executable(
    rocjitsu_benchmark_hipblaslt
    src/hipblaslt_workloads.cpp
)
target_compile_features(rocjitsu_benchmark_hipblaslt PRIVATE cxx_std_20)
target_compile_options(
    rocjitsu_benchmark_hipblaslt
    PRIVATE -Wall -Wextra -Wpedantic
)
target_include_directories(
    rocjitsu_benchmark_hipblaslt
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src"
)
target_link_libraries(
    rocjitsu_benchmark_hipblaslt
    PRIVATE hip::host roc::hipblaslt ${CMAKE_DL_LIBS}
)
set_target_properties(
    rocjitsu_benchmark_hipblaslt
    PROPERTIES
        OUTPUT_NAME rocjitsu-benchmark-hipblaslt
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)

set(_rj_benchmark_targets gfx950 gfx1250)
set(_rj_native_source "${CMAKE_CURRENT_SOURCE_DIR}/src/native_workloads.hip")
set(_rj_native_outputs)

foreach(_target IN LISTS _rj_benchmark_targets)
    set(_output "${CMAKE_CURRENT_BINARY_DIR}/rocjitsu-benchmark-native-${_target}")
    add_custom_command(
        OUTPUT "${_output}"
        COMMAND
            "${RJ_BENCHMARK_AMDCXX}" -x hip --offload-arch=${_target}
            --rocm-path=${ROCM_PATH} -std=c++20 -O3 -DNDEBUG
            -DROCJITSU_BENCHMARK_BUILD_TARGET=\"${_target}\" -Wall -Wextra
            -Wpedantic "${_rj_native_source}" -o "${_output}"
        DEPENDS
            "${_rj_native_source}"
        COMMENT "Building native rocjitsu benchmark workloads (${_target})"
        VERBATIM
    )
    string(REPLACE ":" "_" _target_name "${_target}")
    add_custom_target(
        rocjitsu_benchmark_native_${_target_name} ALL
        DEPENDS "${_output}"
    )
    list(APPEND _rj_native_outputs "${_output}")
    if(_target STREQUAL "gfx950")
        set(_rj_native_metadata_executable "${_output}")
    endif()
endforeach()

add_custom_target(rocjitsu_benchmark_workloads DEPENDS ${_rj_native_outputs})
add_dependencies(
    rocjitsu_benchmark_workloads
    rocjitsu_benchmark_hipblaslt
    rocjitsu_bin
    rocjitsu_plugin_throughput_so
    rocjitsu_shared
)

if(BUILD_TESTING)
    find_package(Python3 3.12 REQUIRED COMPONENTS Interpreter)

    add_executable(
        rocjitsu_benchmark_json_test
        runner/tests/json_output_test.cpp
    )
    target_compile_features(rocjitsu_benchmark_json_test PRIVATE cxx_std_20)
    target_include_directories(
        rocjitsu_benchmark_json_test
        PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src"
    )
    add_test(
        NAME RocjitsuBenchmarkJsonOutputTest
        COMMAND rocjitsu_benchmark_json_test
    )
    set_tests_properties(
        RocjitsuBenchmarkJsonOutputTest
        PROPERTIES LABELS "benchmark;unit" TIMEOUT 10
    )
    add_test(
        NAME RocjitsuBenchmarkCatalogParityTest
        COMMAND
            ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/runner/tests/check_catalog_parity.py
            --manifest ${CMAKE_CURRENT_SOURCE_DIR}/suites/nightly.toml
            --native ${_rj_native_metadata_executable}
            --hipblaslt $<TARGET_FILE:rocjitsu_benchmark_hipblaslt>
    )
    set_tests_properties(
        RocjitsuBenchmarkCatalogParityTest
        PROPERTIES LABELS "benchmark;unit" TIMEOUT 10
    )
    add_test(
        NAME RocjitsuBenchmarkRunnerTest
        COMMAND
            ${Python3_EXECUTABLE} -m unittest discover -s
            ${CMAKE_CURRENT_SOURCE_DIR}/runner/tests -v
    )
    set_tests_properties(
        RocjitsuBenchmarkRunnerTest
        PROPERTIES
            LABELS "benchmark;unit"
            TIMEOUT 30
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..
    )
endif()

unset(_rj_benchmark_targets)
unset(_rj_native_outputs)
unset(_rj_native_source)
