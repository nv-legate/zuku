/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <realm.h>

#include "defer.h"
#include "store.h"

using zuku::after;
using zuku::defer;
using zuku::Store;

auto run() {
  Store<int> a = Store<int>::Create();

  auto token = defer([](int& a) { a = 42; }, a);

  auto done = after(token).defer(
      [](int a) { std::cerr << "Got result " << a << std::endl; }, a);

  auto [last] =
      after(done).defer([] { std::cerr << "All done!" << std::endl; });

  return std::move(last);
}

int main(int argc, char** argv) {
  return zuku::program(
      [] {
        std::cerr << "Hello world!" << std::endl;
        return run();
      },
      argc, argv, {.cpus = 1, .network = false});
}
