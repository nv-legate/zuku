/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <realm.h>

#include "defer.h"
#include "mesh.h"
#include "shape.h"
#include "reshard_test.h"

using zuku::DeviceList;
using zuku::ShardedShape;
namespace {

constexpr int64_t kNumIterations = 1;

}

int main(int argc, char** argv) {
  constexpr int64_t kSize = 16;
  return zuku::program(
      [] {
        DeviceList mesh{{.start = 0, .num_devices = 4}};
        ShardedShape source_shape = ShardedShape{
            .type = zuku::SupportedType::S32,
            .sharding = zuku::Sharding{.dims = {{.size = kSize, .sharding = 2},
                                                {
                                                    .size = 1,
                                                    .sharding = 2,
                                                    .permutation = 1,
                                                }},
                                       .devices = mesh}};
        ShardedShape target_shape = ShardedShape{
            .type = zuku::SupportedType::S32,
            .sharding = zuku::Sharding{
                .dims = {{.size = kSize, .sharding = 1},
                         {.size = 1, .sharding = 4, .permutation = 1}},
                .devices = mesh}};
        zuku::run_reshard_test(source_shape, target_shape, mesh, mesh,
                               kNumIterations);
      },
      argc, argv);
}
