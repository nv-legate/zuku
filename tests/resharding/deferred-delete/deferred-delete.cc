/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <realm.h>

#include "defer.h"
#include "mesh.h"
#include "reshard_test.h"
#include "shape.h"

using zuku::DeviceList;
using zuku::ShardedShape;
constexpr int64_t kNumIterations = 4;
constexpr int kNumElements = 512 * 1024 / sizeof(int);

int main(int argc, char** argv) {
  constexpr int64_t kNumSourceDevices = 2;
  constexpr int64_t kNumDestDevices = 2;
  return zuku::program(
      [] {
        DeviceList source_mesh{{.start = 0, .num_devices = kNumSourceDevices}};
        DeviceList disjoint_target_mesh{
            {.start = kNumSourceDevices, .num_devices = kNumDestDevices}};
        ShardedShape source_shape =
            ShardedShape{.type = zuku::SupportedType::S32,
                         .sharding = zuku::Sharding{
                             .dims = {{.size = kNumElements, .sharding = 1},
                                      {.size = 1,
                                       .sharding = kNumSourceDevices,
                                       .permutation = 1}},
                             .devices = source_mesh}};
        ShardedShape target_shape = ShardedShape{
            .type = zuku::SupportedType::S32,
            .sharding = zuku::Sharding{
                .dims = {{.size = kNumElements, .sharding = kNumDestDevices}},
                .devices = disjoint_target_mesh}};
        zuku::run_reshard_test(source_shape, target_shape, source_mesh,
                               disjoint_target_mesh, kNumIterations);
      },
      argc, argv);
}
