/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "reshard_scatter_gather.h"

#include "gtest/gtest.h"
#include "type_traits.h"

namespace zuku {
namespace {

TEST(ReshardScatterGather, GatherToSingleGPU) {
  static constexpr int kNumSrcDevices = 8;
  static constexpr int kNumDstDevices = 4;
  DeviceList src_mesh{{.start = 0, .num_devices = kNumSrcDevices}};
  DeviceList dst_mesh{{.start = 0, .num_devices = kNumDstDevices}};

  Sharding source_sharding{
      .dims = {ShardingDim{
                   .size = 8,
                   .sharding = 8,
                   .permutation = 0,
               },
               ShardingDim{.size = 2048, .sharding = 1, .permutation = 1}},
      .devices = src_mesh};

  Sharding dest_sharding{
      .dims = {ShardingDim{
                   .size = 8,
                   .sharding = 1,
                   .permutation = 0,
               },
               ShardingDim{.size = 2048, .sharding = 1, .permutation = 1}},
      .devices = src_mesh};

  auto sg_config = GetScatterGatherConfig(
      /*dim=*/0, /*src_shard=*/1, /*dst_shard=*/1, source_sharding,
      dest_sharding);

  for (auto& from_src : sg_config.from_source) {
    std::visit(overloaded{[&](auto&& rect) {
                 std::cerr << "src: " << rect << " " << from_src.src_shard
                           << "->" << from_src.dst_shard << std::endl;
               }},
               from_src.bounds);
  }

  for (auto& to_dst : sg_config.to_target) {
    std::visit(overloaded{[&](auto&& rect) {
                 std::cerr << "src: " << rect << " " << to_dst.src_shard << "->"
                           << to_dst.dst_shard << std::endl;
               }},
               to_dst.bounds);
  }
}

}  // namespace
}  // namespace zuku

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}