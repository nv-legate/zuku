/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef _POC_SRC_HASH_H_
#define _POC_SRC_HASH_H_

#include <functional>
#include <vector>

namespace zuku {

inline void hash_combine(std::size_t&) noexcept {}

template <typename T, typename... Ts>
void hash_combine(std::size_t& target, const T& v, Ts&&... vs) noexcept {
  // NOLINTNEXTLINE(readability-magic-numbers): the constants here are meant to
  // be magic...
  target ^= std::hash<T>{}(v) + 0x9e3779b9 + (target << 6) + (target >> 2);
  hash_combine(target, std::forward<Ts>(vs)...);
}

}  // namespace zuku

template <class T>
struct std::hash<std::vector<T>> {
  std::size_t operator()(const std::vector<T>& v) const {
    std::size_t hash = std::hash<std::size_t>{}(v.size());
    for (auto&& t : v) {
      zuku::hash_combine(hash, t);
    }
    return hash;
  }
};

namespace zuku {

template <typename T, typename... Ts>
std::size_t hash(T&& v, Ts&&... vs) {
  std::size_t initial = std::hash<std::decay_t<T>>{}(v);
  hash_combine(initial, std::forward<Ts>(vs)...);
  return initial;
}

}  // namespace zuku

#endif
