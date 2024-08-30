/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef _POC_SRC_RESHARD_SCATTER_SHARDS_H_
#define _POC_SRC_RESHARD_SCATTER_SHARDS_H_

#include "realm.h"
#include "realm/event.h"
#include "realm/processor.h"

#include <queue>

#include "processor.h"
#include "realm_variants.h"
#include "store.h"
#include "tiled_array.h"

namespace zuku {

struct ScatterShardsConfig {
  std::optional<int64_t> dst_shard;
  bool sender{false};
};

ScatterShardsConfig GetScatterShardsConfig(const Processor& p,
                                           const Sharding& src,
                                           const Sharding& dst);

void ReshardScatterShards(const Processor& p, const ScatterShardsConfig& config,
                          View<ShardedArray> src, Store<ShardedArray>& dst);

}  // namespace zuku

#endif  // _POC_SRC_RESHARD_SCATTER_SHARDS_H_
