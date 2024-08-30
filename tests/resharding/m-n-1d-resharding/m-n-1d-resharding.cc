/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <realm.h>

#include "defer.h"
#include "mesh.h"
#include "shape.h"
#include "reshard_test.h"

namespace {

constexpr int64_t kNumIterations = 5;

}

using zuku::DeviceList;
using zuku::ShardedShape;

int main(int argc, char** argv) {
  return zuku::program(
      [] {
        DeviceList source_mesh{{.start = 0, .num_devices = 4}};
        DeviceList disjoint_target_mesh{{.start = 4, .num_devices = 2}};
        DeviceList nondisjoint_target_mesh{{.start = 0, .num_devices = 2}};
        ShardedShape source_shape = ShardedShape{
            .type = zuku::SupportedType::S32,
            .sharding = zuku::Sharding{.dims = {{.size = 8, .sharding = 4}},
                                       .devices = source_mesh}};
        ShardedShape target_shape = ShardedShape{
            .type = zuku::SupportedType::S32,
            .sharding = zuku::Sharding{.dims = {{.size = 8, .sharding = 2}},
                                       .devices = disjoint_target_mesh}};
        zuku::run_reshard_test(source_shape, target_shape, source_mesh,
                               disjoint_target_mesh, kNumIterations);
        zuku::run_reshard_test(source_shape, target_shape, source_mesh,
                               nondisjoint_target_mesh, kNumIterations);
      },
      argc, argv);
}
