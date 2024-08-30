/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef _POC_SRC_TYPE_UTILS_H_
#define _POC_SRC_TYPE_UTILS_H_

#include <vector>
#include <optional>
#include <iostream>

namespace std {

template <class T>
std::ostream& operator<<(std::ostream& os, const optional<T>& t) {
  if (t.has_value()) {
    os << *t;
  } else {
    os << "nullopt";
  }
  return os;
}

template <class T>
std::ostream& operator<<(std::ostream& os, const vector<T>& v) {
  os << "{ ";
  for (const auto& entry : v) {
    os << entry << " ";
  }
  os << "}";
  return os;
}

}  // namespace std

#endif
