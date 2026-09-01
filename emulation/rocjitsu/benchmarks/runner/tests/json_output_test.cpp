// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "json_output.h"

#include <limits>
#include <sstream>

int main() {
  std::ostringstream output;
  rocjitsu::benchmark::write_json_number(output, 1.5);
  output << ',';
  rocjitsu::benchmark::write_json_number(
      output, std::numeric_limits<double>::infinity());
  output << ',';
  rocjitsu::benchmark::write_json_number(
      output, -std::numeric_limits<double>::infinity());
  output << ',';
  rocjitsu::benchmark::write_json_number(
      output, std::numeric_limits<double>::quiet_NaN());
  return output.str() == "1.5,null,null,null" ? 0 : 1;
}
