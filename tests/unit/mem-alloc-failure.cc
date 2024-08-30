/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <realm.h>
#include <limits>

#include "defer.h"
#include "processor.h"
#include "shape.h"
#include "tiled_array.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using zuku::Processor;
using zuku::ShardedArray;
using zuku::ShardedShape;
using zuku::Store;

Realm::Logger log_test("mem-alloc-failure");

namespace {

void AllocLoop(int num_allocs) {
  std::vector<Store<ShardedArray>> arrays;
  ShardedShape shape{zuku::SupportedType::F64,
                     {
                         .dims = {{
                             .size = int64_t(1e6),
                             .sharding = 1,
                         }},
                         .devices = {{.start = 0, .num_devices = 1}},
                     }};
  for (int i = 0; i < num_allocs; ++i) {
    // pre-allocating would allocate a huge amount of host memory
    // NOLINTNEXTLINE(performance-inefficient-vector-operation)
    arrays.push_back(
        ShardedArray::Create(shape, {.processor = Processor::Default()}));
  }
}

TEST(MemAllocFailureTest, AllocLoop) {
  AllocLoop(10);  // run a small loop to make sure this succeeds
  EXPECT_THROW(AllocLoop(std::numeric_limits<int>::max()),
               zuku::AllocationFailure);
}

}  // namespace

int main(int argc, char** argv) {
  return zuku::program(
      [&] {
        ::testing::InitGoogleTest(&argc, argv);
        return RUN_ALL_TESTS();
      },
      argc, argv);
}
