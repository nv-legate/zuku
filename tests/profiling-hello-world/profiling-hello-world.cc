/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <realm.h>

#include <chrono>
#include <thread>

#include "defer.h"
#include "mesh.h"
#include "processor.h"

using zuku::DeviceList;
using zuku::on;
using zuku::Processor;
using zuku::ProcessorGroup;
using zuku::region;

static constexpr int num_cpus = 2;

auto run() {
  DeviceList mesh{{.start = 0, .num_devices = num_cpus}};
  auto waiter = on(ProcessorGroup::Local()).defer([=](Processor p) {
    auto [result] = region("create").on(p).defer([=] {
      auto [result] = on(p).defer(
          [](int offset) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            return 42 + offset;
          },
          p.local_id());
      return std::move(result);
    });
    auto done = on(p).region("analyze").defer(
        [](int x) {
          std::cerr << "Got result " << x << std::endl;
          std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        },
        std::move(result));
    return done;
  });

  return std::move(waiter);
}

int main(int argc, char** argv) {
  return zuku::program(
      [] {
        std::cerr << "Hello world!" << std::endl;
        return run();
      },
      argc, argv, {.cpus = num_cpus, .network = false, .profile = true});
}
