/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef _POC_SRC_RESHARD_H_
#define _POC_SRC_RESHARD_H_

#include "realm/logging.h"

#include "tiled_array.h"
#include "reshard_gather_shards.h"
#include "reshard_scatter_gather.h"
#include "reshard_scatter_shards.h"
#include "reshard_broadcast.h"

namespace zuku {

extern Realm::Logger log_reshard;

struct PointToPointConfig {};

struct UnsupportedConfig {};

struct NoOpConfig {};

using ReshardConfigVariant =
    std::variant<UnsupportedConfig, PointToPointConfig, ScatterGatherConfig,
                 BroadcastConfig, GatherShardsConfig, ScatterShardsConfig,
                 NoOpConfig>;

ReshardConfigVariant DetermineShardingConfig(const Processor& p,
                                             const Sharding& src,
                                             const Sharding& dst);

void Reshard(Processor p, zuku::View<ShardedArray> src,
             zuku::Store<ShardedArray>& dst);

void Reshard(Processor p, zuku::Store<ShardedArray>& src,
             zuku::Store<ShardedArray>& dst);

}  // namespace zuku

#endif
