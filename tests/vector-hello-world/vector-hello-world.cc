/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <realm.h>

#include "defer.h"
#include "vector.h"

using zuku::after;
using zuku::defer;
using zuku::Future;
using zuku::View;

auto run() {
  int a = 4;
  auto [x, y] = defer([](int a) { return std::make_tuple(a * a, a + a); }, a);

  std::vector<Future<int>> inputs;
  inputs.push_back(std::move(x));
  inputs.push_back(std::move(y));

  auto [sum] = defer(
      [](std::vector<int> values) {
        std::cerr << "Got inputs " << values[0] << " and " << values[1]
                  << std::endl;
        return values[0] + values[1];
      },
      std::move(inputs));

  std::vector<View<int>> ro_views;
  ro_views.push_back(sum.view());
  ro_views.push_back(sum.view());
  auto token = defer(
      [](zuku::ro_vector<int> views) {
        std::cerr << "Here I am!" << std::endl;
        for (const int& val : views) {
          std::cerr << "Got val " << val << std::endl;
        }
      },
      std::move(ro_views));

  auto v = after(token).defer(
      [](int a) { std::cerr << "Got result " << a << std::endl; },
      std::move(sum));  //.view());

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
