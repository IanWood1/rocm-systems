# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Run a HIP workload to normal process exit and verify that the interposer
# stopped its simulator worker cleanly enough to flush exactly one throughput
# summary. execute_process's timeout also turns the ROCr teardown spin that this
# test guards against into a deterministic test failure.
foreach(_required IN ITEMS TEST_EXECUTABLE TEST_FILTER TEST_LAUNCHER TEST_CONFIG TEST_SINK_DIR)
    if(NOT DEFINED ${_required} OR "${${_required}}" STREQUAL "")
        message(FATAL_ERROR "${_required} is required")
    endif()
endforeach()

file(MAKE_DIRECTORY "${TEST_SINK_DIR}")
set(_throughput_log "${TEST_SINK_DIR}/throughput.log")
file(REMOVE "${_throughput_log}")

execute_process(
    COMMAND
        "${CMAKE_COMMAND}" -E env "RJ_SINK_DIR=${TEST_SINK_DIR}"
        "${TEST_LAUNCHER}" --config "${TEST_CONFIG}" --
        "${TEST_EXECUTABLE}" "--gtest_filter=${TEST_FILTER}"
    RESULT_VARIABLE _result
    OUTPUT_VARIABLE _stdout
    ERROR_VARIABLE _stderr
    TIMEOUT 20
)

if(NOT "${_result}" STREQUAL "0")
    message(
        FATAL_ERROR
        "HIP process did not exit normally (result: ${_result})\n"
        "stdout:\n${_stdout}\n"
        "stderr:\n${_stderr}"
    )
endif()

if(NOT EXISTS "${_throughput_log}")
    message(FATAL_ERROR "throughput log was not created: ${_throughput_log}")
endif()

file(READ "${_throughput_log}" _throughput_output)
string(
    REGEX MATCHALL
    "\"record\"[ \t]*:[ \t]*\"summary\""
    _summaries
    "${_throughput_output}"
)
list(LENGTH _summaries _summary_count)
if(NOT _summary_count EQUAL 1)
    message(
        FATAL_ERROR
        "expected exactly one throughput summary, found ${_summary_count}\n"
        "throughput.log:\n${_throughput_output}"
    )
endif()

string(
    REGEX MATCHALL
    "\"record\"[ \t]*:[ \t]*\"dispatch\""
    _dispatches
    "${_throughput_output}"
)
list(LENGTH _dispatches _dispatch_count)
if(_dispatch_count LESS 1)
    message(
        FATAL_ERROR
        "expected at least one throughput dispatch record\n"
        "throughput.log:\n${_throughput_output}"
    )
endif()

message(
    STATUS
    "HIP process exited normally; throughput log has "
    "${_dispatch_count} dispatch record(s) and exactly one summary"
)
