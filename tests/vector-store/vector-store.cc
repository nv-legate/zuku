/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <realm.h>

#include "defer.h"
#include "store.h"
#include "task.h"

using zuku::defer;
using zuku::Store;

auto run() {
  Store<int> a = Store<int>::Create();
  Store<int> b = Store<int>::Create();

  zuku::rw_vector<Store<int>> stores{a, b};

  defer(
      [](zuku::rw_vector<int> vals) {
        vals[0] = 42;
        vals[1] = 43;
      },
      stores);

  zuku::store_variant_vector<int> vars;
  vars.push_back(a.view());
  vars.push_back(b);

  defer(
      [](zuku::ro_vector<int> vals) {
        std::cerr << vals[0] << " and " << vals[1] << std::endl;
      },
      std::move(vars));

  defer(
      [](zuku::rw_vector<int> vals) {
        vals[0] = 41;
        vals[1] = 39;
      },
      stores);

  return defer(
      [](zuku::ro_vector<int> vals) {
        std::cerr << vals[0] << " and " << vals[1] << std::endl;
      },
      stores);
}

int main(int argc, char** argv) {
  return zuku::program(
      [] {
        std::cerr << "Hello world!" << std::endl;
        return run();
      },
      argc, argv, {.cpus = 1, .network = false});
}
