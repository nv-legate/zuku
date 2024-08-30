/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "type_traits.h"

#include "gtest/gtest.h"
#include "tiled_array.h"

namespace zuku {
namespace {

TEST(TypeTraitsTest, ValidateStoreTraits) {
  static_assert(has_deferred_delete_instances<ShardedArray>::value,
                "Store does not have a DeferredDeleteInstance method");
};

}  // namespace
}  // namespace zuku

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}