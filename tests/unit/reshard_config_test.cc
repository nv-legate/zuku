/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <variant>
#include "processor.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "reshard.h"
#include "reshard_gather_shards.h"
#include "reshard_scatter_gather.h"
#include "tiled_array.h"
#include "shape_utils.h"

namespace zuku {
namespace {

using ::testing::Each;
using ::testing::Eq;

template <int N, typename T>
void Cover(std::vector<int>& entries, const Realm::Rect<N, T>& rect) {
  if constexpr (N == 1) {
    for (int i = rect.lo; i <= rect.hi; ++i) {
      entries[i] = 1;
    }
  }
}

void Cover(std::vector<int>& entries, const RealmShape& shape) {
  std::visit(overloaded{[&](const auto& rect) { Cover(entries, rect); }},
             shape);
}

TEST(ReshardConfigTest, ValidatePointToPoint) {
  Sharding src{
      .dims = {{
                   .size = 16,
                   .sharding = 4,
                   .permutation = 0,
               },
               {.size = 16, .sharding = 2, .permutation = 1}},
      .devices = {{.start = 0, .num_devices = 8}},
  };

  Sharding dst{
      .dims = src.dims,
      .devices = {{.start = 8, .num_devices = 8}},
  };

  Processor p =
      Processor::Create({.local = 0, .global = 0}, Processor::Type::TEST);

  auto config = DetermineShardingConfig(p, src, dst);
  EXPECT_TRUE(std::holds_alternative<PointToPointConfig>(config));
};

TEST(ReshardConfigTest, ValidateScatterGather) {
  Sharding src{
      .dims = {{
                   .size = 16,
                   .sharding = 2,
                   .permutation = 0,
               },
               {.size = 16, .sharding = 2, .permutation = 1}},
      .devices = {{.start = 0, .num_devices = 4}},
  };

  Sharding dst{
      .dims = {{
                   .size = 16,
                   .sharding = 2,
                   .permutation = 0,
               },
               {.size = 16, .sharding = 4, .permutation = 1}},
      .devices = {{.start = 4, .num_devices = 8}},
  };

  Processor p =
      Processor::Create({.local = 0, .global = 0}, Processor::Type::TEST);

  auto config = DetermineShardingConfig(p, src, dst);
  ASSERT_TRUE(std::holds_alternative<ScatterGatherConfig>(config));
  const ScatterGatherConfig& sg_config = std::get<ScatterGatherConfig>(config);

  int64_t source_size = 0;
  for (const auto& entry : sg_config.from_source) {
    source_size += ComputeRealmShapeSize(entry.bounds);
  }
  EXPECT_EQ(source_size, ShardElements(src));
};

TEST(ReshardConfigTest, ValidateReplicatedGather) {
  constexpr int64_t kSize = 16;
  Sharding src{
      .dims = {{
                   .size = kSize,
                   .sharding = 2,
                   .permutation = 0,
               },
               {.size = 1, .sharding = 2, .permutation = 1}},
      .devices = {{.start = 0, .num_devices = 4}},
  };

  Sharding dst{
      .dims = {{
                   .size = kSize,
                   .sharding = 1,
                   .permutation = 0,
               },
               {.size = 1, .sharding = 8, .permutation = 1}},
      .devices = {{.start = 0, .num_devices = 8}},
  };

  std::vector<int> send_covered(kSize, 0);

  for (int i = 0; i < 4; ++i) {
    Processor p =
        Processor::Create({.local = i, .global = i}, Processor::Type::TEST);
    auto config = DetermineShardingConfig(p, src, dst);
    ASSERT_TRUE(std::holds_alternative<GatherShardsConfig>(config));
    const GatherShardsConfig& g_config = std::get<GatherShardsConfig>(config);
    ASSERT_TRUE(g_config.source_to_target_bounds.has_value());
    Cover(send_covered, *g_config.source_to_target_bounds);
  }
  EXPECT_THAT(send_covered, Each(Eq(1)));

  for (int i = 0; i < 8; ++i) {
    Processor p =
        Processor::Create({.local = i, .global = i}, Processor::Type::TEST);
    auto config = DetermineShardingConfig(p, src, dst);
    ASSERT_TRUE(std::holds_alternative<GatherShardsConfig>(config));
    const GatherShardsConfig& g_config = std::get<GatherShardsConfig>(config);
    std::vector<int> recv_covered(kSize, 0);
    for (const auto& slice : g_config.target_from_source_bounds) {
      Cover(recv_covered, slice);
    }
    EXPECT_THAT(recv_covered, Each(Eq(1)));
  }
}

TEST(ReshardConfigTest, ValidateReplicatedGatherNonParticipants) {
  constexpr int64_t kSize = 16;
  Sharding src{
      .dims = {{
                   .size = kSize,
                   .sharding = 2,
                   .permutation = 0,
               },
               {.size = 1, .sharding = 2, .permutation = 1}},
      .devices = {{.start = 0, .num_devices = 4}},
  };

  Sharding dst{
      .dims = {{
          .size = kSize,
          .sharding = 1,
          .permutation = 0,
      }},
      .devices = {{.start = 0, .num_devices = 4}},
  };

  Processor p =
      Processor::Create({.local = 4, .global = 4}, Processor::Type::TEST);
  auto config = DetermineShardingConfig(p, src, dst);
  ASSERT_TRUE(std::holds_alternative<GatherShardsConfig>(config));
  const GatherShardsConfig& g_config = std::get<GatherShardsConfig>(config);
}

void ChangeAxisShardingOn3D(const Sharding& src, const Sharding& dst,
                            int nproc) {
  for (int i = 0; i < nproc; ++i) {
    Processor p =
        Processor::Create({.local = i, .global = i}, Processor::Type::TEST);
    auto config = DetermineShardingConfig(p, src, dst);
    ASSERT_TRUE(std::holds_alternative<ScatterGatherConfig>(config));
    const ScatterGatherConfig& s_config = std::get<ScatterGatherConfig>(config);
    if (src.devices.Contains(i)) {
      auto source_bounds = ComputeTileBounds(i, src);
      const int64_t box_size = ComputeRealmShapeSize(source_bounds);
      std::vector<int> covered(box_size, 0);
      for (const auto& slice : s_config.from_source) {
        EXPECT_EQ(slice.src_shard, i);
        zuku::Iterate(covered.data(), slice.bounds, source_bounds,
                      [](int& data, auto... indices) { data = 1; });
      }
      EXPECT_THAT(covered, Each(Eq(1)));
    } else {
      EXPECT_TRUE(s_config.from_source.empty());
    }

    if (dst.devices.Contains(i)) {
      auto dest_bounds = ComputeTileBounds(i, dst);
      const int64_t box_size = ComputeRealmShapeSize(dest_bounds);
      std::vector<int> covered(box_size, 0);
      for (const auto& slice : s_config.to_target) {
        EXPECT_EQ(slice.dst_shard, i);
        zuku::Iterate(covered.data(), slice.bounds, dest_bounds,
                      [](int& data, auto... indices) { data = 1; });
      }
      EXPECT_THAT(covered, Each(Eq(1)));
    } else {
      EXPECT_TRUE(s_config.to_target.empty());
    }
  }
}

TEST(ReshardConfigTest, PartialToReplicated1D) {
  DeviceList mesh{{.start = 0, .num_devices = 4}};
  constexpr int64_t kSize = 16;
  Sharding src = {.dims = {{.size = kSize, .sharding = 2},
                           {
                               .size = 1,
                               .sharding = 2,
                               .permutation = 1,
                           }},
                  .devices = mesh};
  Sharding dst = {.dims = {{.size = kSize, .sharding = 1}}, .devices = mesh};

  for (int i = 0; i < 4; ++i) {
    Processor p =
        Processor::Create({.local = i, .global = i}, Processor::Type::TEST);
    auto config = DetermineShardingConfig(p, src, dst);
    ASSERT_TRUE(std::holds_alternative<GatherShardsConfig>(config));
    const GatherShardsConfig& g_config = std::get<GatherShardsConfig>(config);
  }
}

TEST(ReshardConfigTest, PartialReplicatedToReplicated3D) {
  Sharding src{
      .dims = {{
                   .size = 8,
                   .sharding = 2,
                   .permutation = 0,
               },
               {.size = 128, .sharding = 1, .permutation = 1},
               {.size = 256, .sharding = 2, .permutation = 2},
               {.size = 1, .sharding = 2, .permutation = 3}},
      .devices = {{.start = 0, .num_devices = 8}},
  };

  Sharding dst{
      .dims =
          {
              {
                  .size = 8,
                  .sharding = 1,
                  .permutation = 0,
              },
              {.size = 128, .sharding = 1, .permutation = 1},
              {.size = 256, .sharding = 1, .permutation = 2},
          },
      .devices = {{.start = 0, .num_devices = 4}},
  };

  for (int i = 0; i < 8; ++i) {
    Processor p =
        Processor::Create({.local = i, .global = i}, Processor::Type::TEST);
    auto config = DetermineShardingConfig(p, src, dst);
    ASSERT_TRUE(std::holds_alternative<GatherShardsConfig>(config));
    const GatherShardsConfig& g_config = std::get<GatherShardsConfig>(config);
  }
}

TEST(ReshardConfigTest, Partial2x2ToReplicated2D) {
  constexpr int64_t kSize = 16;
  Sharding src{
      .dims = {{
                   .size = 256,
                   .sharding = 2,
                   .permutation = 1,
               },
               {.size = 1, .sharding = 2, .permutation = 0}},
      .devices = {{.start = 0, .num_devices = 4}},
  };

  Sharding dst{
      .dims = {{
                   .size = 256,
                   .sharding = 1,
                   .permutation = 0,
               },
               {.size = 1, .sharding = 4, .permutation = 1}},
      .devices = {{.start = 0, .num_devices = 4}},
  };

  for (int i = 0; i < 4; ++i) {
    Processor p =
        Processor::Create({.local = i, .global = i}, Processor::Type::TEST);
    auto config = DetermineShardingConfig(p, src, dst);
    ASSERT_TRUE(std::holds_alternative<GatherShardsConfig>(config));
    const GatherShardsConfig& g_config = std::get<GatherShardsConfig>(config);
    auto bounds = ComputeTileBounds(i, src);
    // make sure that all source shards are contined within the original rect
    ASSERT_TRUE(g_config.source_to_target_bounds.has_value());
    EXPECT_TRUE(ShapeContains(bounds, *g_config.source_to_target_bounds));
  }
}

TEST(ReshardConfigTest, IncreaseShardingOnAxis3D) {
  constexpr int64_t kSize = 16;
  Sharding src{
      .dims = {{
                   .size = 8,
                   .sharding = 2,
                   .permutation = 0,
               },
               {.size = 128, .sharding = 1, .permutation = 1},
               {.size = 256, .sharding = 2, .permutation = 2}},
      .devices = {{.start = 0, .num_devices = 4}},
  };

  Sharding dst{
      .dims = {{
                   .size = 8,
                   .sharding = 4,
                   .permutation = 0,
               },
               {.size = 128, .sharding = 1, .permutation = 1},
               {.size = 256, .sharding = 2, .permutation = 2}},
      .devices = {{.start = 0, .num_devices = 8}},
  };
  ChangeAxisShardingOn3D(src, dst, 8);
}

TEST(ReshardConfigTest, DecreaseShardingOnAxis3D) {
  constexpr int64_t kSize = 16;
  Sharding src{
      .dims = {{
                   .size = 8,
                   .sharding = 4,
                   .permutation = 0,
               },
               {.size = 128, .sharding = 1, .permutation = 1},
               {.size = 256, .sharding = 2, .permutation = 2}},
      .devices = {{.start = 0, .num_devices = 8}},
  };

  Sharding dst{
      .dims = {{
                   .size = 8,
                   .sharding = 2,
                   .permutation = 0,
               },
               {.size = 128, .sharding = 1, .permutation = 1},
               {.size = 256, .sharding = 2, .permutation = 2}},
      .devices = {{.start = 0, .num_devices = 4}},
  };
  ChangeAxisShardingOn3D(src, dst, 8);
}

TEST(ReshardConfigTest, Expand3DSharding) {
  Sharding src{
      .dims = {{
                   .size = 8,
                   .sharding = 2,
                   .permutation = 0,
               },
               {.size = 128, .sharding = 1, .permutation = 1},
               {.size = 32, .sharding = 2, .permutation = 2}},
      .devices = {{.start = 4, .num_devices = 4}},
  };

  Sharding dst{
      .dims = {{
                   .size = 8,
                   .sharding = 4,
                   .permutation = 0,
               },
               {.size = 128, .sharding = 1, .permutation = 1},
               {.size = 32, .sharding = 2, .permutation = 2}},
      .devices = {{.start = 0, .num_devices = 8}},
  };

  for (int i = 0; i < 8; ++i) {
    Processor p =
        Processor::Create({.local = i, .global = i}, Processor::Type::TEST);
    auto config = DetermineShardingConfig(p, src, dst);
    ASSERT_TRUE(std::holds_alternative<ScatterGatherConfig>(config));
    const ScatterGatherConfig& sg_config =
        std::get<ScatterGatherConfig>(config);

    auto tile_bounds = ComputeTileBounds(i, dst);
    int64_t tile_size = ComputeRealmShapeSize(tile_bounds);
    std::vector<int> covered(tile_size, 0);
    // make sure the dst is covered
    for (const auto& slice : sg_config.to_target) {
      zuku::Iterate(covered.data(), slice.bounds, tile_bounds,
                    [](int& data, auto... indices) { data = 1; });
    }
    EXPECT_THAT(covered, Each(Eq(1)));
  }
}

TEST(ReshardConfigTest, Contract3DSharding) {
  Sharding dst{
      .dims = {{
                   .size = 8,
                   .sharding = 2,
                   .permutation = 0,
               },
               {.size = 128, .sharding = 1, .permutation = 1},
               {.size = 32, .sharding = 2, .permutation = 2}},
      .devices = {{.start = 4, .num_devices = 4}},
  };

  Sharding src{
      .dims = {{
                   .size = 8,
                   .sharding = 4,
                   .permutation = 0,
               },
               {.size = 128, .sharding = 1, .permutation = 1},
               {.size = 32, .sharding = 2, .permutation = 2}},
      .devices = {{.start = 0, .num_devices = 8}},
  };

  for (int i = 0; i < 8; ++i) {
    Processor p =
        Processor::Create({.local = i, .global = i}, Processor::Type::TEST);
    auto config = DetermineShardingConfig(p, src, dst);
    ASSERT_TRUE(std::holds_alternative<ScatterGatherConfig>(config));
    const ScatterGatherConfig& sg_config =
        std::get<ScatterGatherConfig>(config);

    if (i >= 4) {
      auto tile_bounds = ComputeTileBounds(i - 4, dst);
      int64_t tile_size = ComputeRealmShapeSize(tile_bounds);
      std::vector<int> covered(tile_size, 0);
      // make sure the dst is covered
      for (const auto& slice : sg_config.to_target) {
        zuku::Iterate(covered.data(), slice.bounds, tile_bounds,
                      [](int& data, auto... indices) { data = 1; });
      }
      EXPECT_THAT(covered, Each(Eq(1)));
    }
  }
}

TEST(ReshardConfigTest, Bad3DReshardingWithPartialReplication) {
  Sharding src{
      .dims = {{
                   .size = 8,
                   .sharding = 4,
                   .permutation = 0,
               },
               {.size = 128, .sharding = 1, .permutation = 1},
               {.size = 1, .sharding = 1, .permutation = 2},
               {.size = 1, .sharding = 2, .permutation = 3}},
      .devices = {{.start = 0, .num_devices = 8}},
  };

  Sharding dst{
      .dims = {{
                   .size = 8,
                   .sharding = 2,
                   .permutation = 0,
               },
               {.size = 128, .sharding = 1, .permutation = 1},
               {.size = 1, .sharding = 1, .permutation = 2},
               {.size = 1, .sharding = 2, .permutation = 3}},
      .devices = {{.start = 0, .num_devices = 4}},
  };

  for (int i = 0; i < 8; ++i) {
    Processor p =
        Processor::Create({.local = i, .global = i}, Processor::Type::TEST);
    auto config = DetermineShardingConfig(p, src, dst);
    ASSERT_TRUE(std::holds_alternative<ScatterGatherConfig>(config));
    const ScatterGatherConfig& sg_config =
        std::get<ScatterGatherConfig>(config);

    if (i < 4) {
      auto tile_bounds = ComputeTileBounds(i, dst);
      int64_t tile_size = ComputeRealmShapeSize(tile_bounds);
      std::vector<int> covered(tile_size, 0);
      // make sure the dst is covered
      for (const auto& slice : sg_config.to_target) {
        zuku::Iterate(covered.data(), slice.bounds, tile_bounds,
                      [](int& data, auto... indices) { data = 1; });
      }
      EXPECT_THAT(covered, Each(Eq(1)));
    }
  }
}

}  // namespace
}  // namespace zuku

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}