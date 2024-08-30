/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <realm.h>

#include "defer.h"
#include "type_traits.h"

using zuku::after;
using zuku::defer;

auto run() {
  int a = 4;
  auto [x] = defer([](int a) { return a * a; }, a);

  auto [y] = defer([] { return defer([] { return 42; }).futures(); });

  auto x_view = x.view();
  auto v = defer(
      [](int a, const int& b, zuku::View<int> c) {
        std::cerr << "Got results " << a << " and " << b << std::endl;
        defer([](int c) { std::cerr << "Got third result " << c << std::endl; },
              std::move(c));
      },
      x, std::move(y), x_view);

  auto [waiter] = after(v).defer([] { std::cerr << "All done!" << std::endl; });

  return std::move(waiter);
}

int main(int argc, char** argv) {
  return zuku::program(
      [] {
        std::cerr << "Hello world!" << std::endl;
        return run();
      },
      argc, argv, {.cpus = 1, .network = false});
}
