// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_BENCHMARK_JSON_OUTPUT_H
#define ROCJITSU_BENCHMARK_JSON_OUTPUT_H

#include <cmath>
#include <ostream>

namespace rocjitsu::benchmark {

inline void write_json_number(std::ostream &output, double value) {
  if (std::isfinite(value))
    output << value;
  else
    output << "null";
}

} // namespace rocjitsu::benchmark

#endif // ROCJITSU_BENCHMARK_JSON_OUTPUT_H
