/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef _POC_SRC_RESHARD_SCATTER_GATHER_H_
#define _POC_SRC_RESHARD_SCATTER_GATHER_H_

#include "realm.h"
#include "realm/event.h"
#include "realm/processor.h"

#include <queue>

#include "realm_variants.h"
#include "store.h"
#include "tiled_array.h"

namespace zuku {

struct SliceConfig {
  const int64_t src_shard;
  const int64_t dst_shard;
  RealmShape bounds;
};

std::ostream& operator<<(std::ostream& os, const SliceConfig& config);

struct ScatterGatherConfig {
  std::vector<SliceConfig> from_source;
  std::vector<SliceConfig> to_target;
};

std::ostream& operator<<(std::ostream& os, const ScatterGatherConfig& config);

RealmShape MakeTileBox(const std::vector<int64_t>& index,
                       const Sharding& sharding);

ScatterGatherConfig GetScatterGatherConfig(int64_t dim,
                                           std::optional<int64_t> src_shard,
                                           std::optional<int64_t> dst_shard,
                                           const Sharding& src,
                                           const Sharding& dst);

void ReshardScatterGather(const Processor& p, const ScatterGatherConfig& config,
                          View<zuku::ShardedArray> src,
                          Store<zuku::ShardedArray>& dst);

}  // namespace zuku

#endif
