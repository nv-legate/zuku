/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef _POC_SRC_RESHARD_GATHER_SHARDS_H_
#define _POC_SRC_RESHARD_GATHER_SHARDS_H_

#include "realm.h"

#include "processor.h"
#include "realm_variants.h"
#include "store.h"
#include "tiled_array.h"

namespace zuku {

struct GatherShardsConfig {
  std::optional<int64_t> src_shard;
  std::optional<int64_t> dst_shard;
  std::optional<RealmShape> source_to_target_bounds;
  std::vector<RealmShape> target_from_source_bounds;
};

std::ostream& operator<<(std::ostream& os, const GatherShardsConfig& config);

bool operator==(const GatherShardsConfig& lhs, const GatherShardsConfig& rhs);

GatherShardsConfig GetGatherShardsConfig(const Processor& p,
                                         const Sharding& src,
                                         const Sharding& dst);

void ReshardGatherShards(const Processor& p, const GatherShardsConfig& config,
                         View<ShardedArray> src, Store<ShardedArray>& dst);

}  // namespace zuku

#endif  // _POC_SRC_RESHARD_GATHER_SHARDS_H_
