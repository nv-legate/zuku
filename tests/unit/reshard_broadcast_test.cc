/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "reshard_broadcast.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace zuku {
namespace {

using ::testing::ElementsAre;
using ::testing::ElementsAreArray;

Sharding GetReplicatedSharding(DeviceList devices) {
  return Sharding{.dims = {ShardingDim{
                      .size = 8,
                      .sharding = 1,
                      .permutation = 0,
                  }},
                  .devices = std::move(devices)};
}

TEST(ReshardBroadcast, RootConfigTestPowerOf2) {
  static constexpr int kNumSrcDevices = 2;
  static constexpr int kNumDstDevices = 8;
  DeviceList src_mesh{{.start = 0, .num_devices = kNumSrcDevices}};
  DeviceList dst_mesh{{.start = 0, .num_devices = kNumDstDevices}};

  auto src = GetBroadcastConfig(/*src_shard=*/0, std::nullopt,
                                GetReplicatedSharding(src_mesh),
                                GetReplicatedSharding(dst_mesh));

  ASSERT_TRUE(src.source.has_value());
  EXPECT_THAT(src.source->dst_root_shards, ElementsAre(0));
  EXPECT_FALSE(src.target.has_value());

  // 0 ---> 0 -> 2 -> 1
  // 1 ->
  // 2 -> 3
  // 3 ->
  // 4 ----> 4 -> 6 -> 5
  // 5 ->
  // 6 -> 7
  // 7 ->
  std::vector<BroadcastConfig> expected = {
      {.source =
           SourceConfig{
               .dst_root_shards = {0},
           },
       .target = TargetConfig{.src_shard = 0, .dst_leaf_shards = {2, 1}}},
      {.source =
           SourceConfig{
               .dst_root_shards = {4},
           },
       .target = TargetConfig{.dst_root_shard = 0}},
      {.target =
           TargetConfig{
               .dst_root_shard = 0,
               .dst_leaf_shards = {3},
           }},
      {.target = TargetConfig{.dst_root_shard = 2}},
      {.target =
           TargetConfig{
               .src_shard = 1,
               .dst_leaf_shards = {6, 5},
           }},
      {.target =
           TargetConfig{
               .dst_root_shard = 4,
           }},
      {.target =
           TargetConfig{
               .dst_root_shard = 4,
               .dst_leaf_shards = {7},
           }},
      {.target = TargetConfig{.dst_root_shard = 6}},
  };

  std::vector<BroadcastConfig> test;
  for (int i = 0; i < kNumDstDevices; ++i) {
    auto src_shard = [&]() -> std::optional<int64_t> {
      if (i < kNumSrcDevices) {
        return i;
      }
      return std::nullopt;
    }();
    test.push_back(GetBroadcastConfig(src_shard, /*dst_shard=*/i,
                                      GetReplicatedSharding(src_mesh),
                                      GetReplicatedSharding(dst_mesh)));
  }

  EXPECT_THAT(test, ElementsAreArray(expected));
}

TEST(ReshardBroadcast, RootConfigTestUneven) {
  static constexpr int kNumSrcDevices = 3;
  static constexpr int kNumDstDevices = 7;
  DeviceList src_mesh{{.start = 0, .num_devices = kNumSrcDevices}};
  DeviceList dst_mesh{{.start = 0, .num_devices = kNumDstDevices}};

  auto src1 = GetBroadcastConfig(/*src_shard=*/1, std::nullopt,
                                 GetReplicatedSharding(src_mesh),
                                 GetReplicatedSharding(dst_mesh));

  ASSERT_TRUE(src1.source.has_value());
  EXPECT_THAT(src1.source->dst_root_shards, ElementsAre(3));
  EXPECT_FALSE(src1.target.has_value());

  // 0 ---> 0 -> 2 -> 1
  // 1 ---> 3 -> 5 -> 4
  // 2 ---> 6
  auto dst4 = GetBroadcastConfig(/*src_shard=*/std::nullopt, /*dst_shard=*/4,
                                 GetReplicatedSharding(src_mesh),
                                 GetReplicatedSharding(dst_mesh));
  ASSERT_TRUE(dst4.target.has_value());
  ASSERT_TRUE(dst4.target->dst_root_shard.has_value());
  EXPECT_FALSE(dst4.target->src_shard.has_value());
  EXPECT_EQ(*dst4.target->dst_root_shard, 3);
  EXPECT_TRUE(dst4.target->dst_leaf_shards.empty());

  auto dst6 = GetBroadcastConfig(/*src_shard=*/std::nullopt, /*dst_shard=*/6,
                                 GetReplicatedSharding(src_mesh),
                                 GetReplicatedSharding(dst_mesh));
  ASSERT_TRUE(dst6.target.has_value());
  ASSERT_TRUE(dst6.target->src_shard.has_value());
  EXPECT_EQ(*dst6.target->src_shard, 2);
  EXPECT_FALSE(dst6.target->dst_root_shard.has_value());
  EXPECT_TRUE(dst6.target->dst_leaf_shards.empty());
}

}  // namespace
}  // namespace zuku

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}