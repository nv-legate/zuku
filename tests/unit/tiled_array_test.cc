/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "tiled_array.h"

#include <span>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "mesh.h"

namespace zuku {
namespace {

using ::testing::ElementsAre;

template <int N, class T>
std::span<const T> Span(const Realm::Point<N, T>& pt) {
  return {&pt[0], &pt[0] + N};
}

TEST(TiledArrayTest, ComputeTileIndexFullySharded) {
  DeviceList devices{{.start = 0, .num_devices = 6}};
  Sharding sharding{.dims = {{.size = 8, .sharding = 2, .permutation = 0},
                             {.size = 9, .sharding = 3, .permutation = 1}},
                    .devices = devices};

  EXPECT_THAT(ComputeTileIndex(0, sharding), ElementsAre(0, 0));
  EXPECT_THAT(ComputeTileIndex(1, sharding), ElementsAre(0, 1));
  EXPECT_THAT(ComputeTileIndex(5, sharding), ElementsAre(1, 2));
}

TEST(TiledArrayTest, ComputeTileIndexPartialPermuted) {
  DeviceList devices{{.start = 0, .num_devices = 6}};
  Sharding sharding{.dims = {{.size = 8, .sharding = 2, .permutation = 2},
                             {.size = 9, .sharding = 3, .permutation = 0},
                             {.size = 1, .sharding = 4, .permutation = 1}},
                    .devices = devices};

  EXPECT_THAT(ComputeTileIndex(0, sharding), ElementsAre(0, 0, 0));
  EXPECT_THAT(ComputeTileIndex(1, sharding), ElementsAre(1, 0, 0));
  EXPECT_THAT(ComputeTileIndex(2, sharding), ElementsAre(0, 0, 1));
  EXPECT_THAT(ComputeTileIndex(3, sharding), ElementsAre(1, 0, 1));
  EXPECT_THAT(ComputeTileIndex(4, sharding), ElementsAre(0, 0, 2));
  EXPECT_THAT(ComputeTileIndex(5, sharding), ElementsAre(1, 0, 2));
  EXPECT_THAT(ComputeTileIndex(11, sharding), ElementsAre(1, 1, 1));
  EXPECT_THAT(ComputeTileIndex(12, sharding), ElementsAre(0, 1, 2));
}

TEST(TiledArrayTest, ComputeTileBoundsFullySharded) {
  DeviceList devices{{.start = 0, .num_devices = 6}};
  Sharding sharding{.dims = {{.size = 8, .sharding = 2, .permutation = 0},
                             {.size = 9, .sharding = 3, .permutation = 1}},
                    .devices = devices};

  // tile 0,0
  std::visit(
      [&](const auto& rect) {
        EXPECT_THAT(Span(rect.hi), ElementsAre(3, 2));
        EXPECT_THAT(Span(rect.lo), ElementsAre(0, 0));
      },
      ComputeTileBounds(0, sharding));

  // tile 0,1
  std::visit(
      [&](const auto& rect) {
        EXPECT_THAT(Span(rect.hi), ElementsAre(3, 5));
        EXPECT_THAT(Span(rect.lo), ElementsAre(0, 3));
      },
      ComputeTileBounds(1, sharding));

  // tile 1,2
  std::visit(
      [&](const auto& rect) {
        EXPECT_THAT(Span(rect.hi), ElementsAre(7, 8));
        EXPECT_THAT(Span(rect.lo), ElementsAre(4, 6));
      },
      ComputeTileBounds(5, sharding));
}

TEST(TiledArrayTest, ComputeTileBoundsPartialReplication) {
  DeviceList devices{{.start = 0, .num_devices = 24}};
  Sharding sharding{.dims = {{.size = 8, .sharding = 2, .permutation = 0},
                             {.size = 1,
                              .sharding = 4,
                              .permutation = 1},  // partial replication
                             {.size = 9, .sharding = 3, .permutation = 2}},
                    .devices = devices};

  // tile 0,0,0
  std::visit(
      [&](const auto& rect) {
        EXPECT_THAT(Span(rect.hi), ElementsAre(3, 2));
        EXPECT_THAT(Span(rect.lo), ElementsAre(0, 0));
      },
      ComputeTileBounds(0, sharding));

  // tile 0,1,1
  std::visit(
      [&](const auto& rect) {
        EXPECT_THAT(Span(rect.hi), ElementsAre(3, 5));
        EXPECT_THAT(Span(rect.lo), ElementsAre(0, 3));
      },
      ComputeTileBounds(4, sharding));

  // tile 1,2,2
  std::visit(
      [&](const auto& rect) {
        EXPECT_THAT(Span(rect.hi), ElementsAre(7, 8));
        EXPECT_THAT(Span(rect.lo), ElementsAre(4, 6));
      },
      ComputeTileBounds(20, sharding));
}

TEST(TiledArrayTest, ComputeTileBoundsPartialReplicationPermuted) {
  DeviceList devices{{.start = 0, .num_devices = 24}};
  Sharding sharding{.dims = {{.size = 8, .sharding = 2, .permutation = 2},
                             {.size = 1,
                              .sharding = 4,
                              .permutation = 0},  // partial replication
                             {.size = 9, .sharding = 3, .permutation = 1}},
                    .devices = devices};

  // tile 0,0,0
  std::visit(
      [&](const auto& rect) {
        EXPECT_THAT(Span(rect.hi), ElementsAre(3, 2));
        EXPECT_THAT(Span(rect.lo), ElementsAre(0, 0));
      },
      ComputeTileBounds(0, sharding));

  // tile 0,0,2 -> 0,2
  std::visit(
      [&](const auto& rect) {
        EXPECT_THAT(Span(rect.hi), ElementsAre(3, 8));
        EXPECT_THAT(Span(rect.lo), ElementsAre(0, 6));
      },
      ComputeTileBounds(4, sharding));

  // tile 0,3,1 -> 0,1
  std::visit(
      [&](const auto& rect) {
        EXPECT_THAT(Span(rect.hi), ElementsAre(3, 5));
        EXPECT_THAT(Span(rect.lo), ElementsAre(0, 3));
      },
      ComputeTileBounds(20, sharding));

  // tile 1,2,1 -> 0,1
  std::visit(
      [&](const auto& rect) {
        EXPECT_THAT(Span(rect.hi), ElementsAre(7, 5));
        EXPECT_THAT(Span(rect.lo), ElementsAre(4, 3));
      },
      ComputeTileBounds(15, sharding));
}

TEST(TiledArrayTest, ComputeTileBoundsFullyReplicated) {
  DeviceList devices{{.start = 0, .num_devices = 4}};
  Sharding sharding{.dims = {{.size = 1, .sharding = 4, .permutation = 0},
                             {.size = 8, .sharding = 1, .permutation = 1},
                             {.size = 6, .sharding = 1, .permutation = 2}},
                    .devices = devices};

  // tile 0,0,0
  std::visit(
      [&](const auto& rect) {
        EXPECT_THAT(Span(rect.hi), ElementsAre(7, 5));
        EXPECT_THAT(Span(rect.lo), ElementsAre(0, 0));
      },
      ComputeTileBounds(0, sharding));

  // tile 2,0,0
  std::visit(
      [&](const auto& rect) {
        EXPECT_THAT(Span(rect.hi), ElementsAre(7, 5));
        EXPECT_THAT(Span(rect.lo), ElementsAre(0, 0));
      },
      ComputeTileBounds(2, sharding));

  // tile 3,0,0
  std::visit(
      [&](const auto& rect) {
        EXPECT_THAT(Span(rect.hi), ElementsAre(7, 5));
        EXPECT_THAT(Span(rect.lo), ElementsAre(0, 0));
      },
      ComputeTileBounds(3, sharding));
}

}  // namespace
}  // namespace zuku

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}