/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "reshard_scatter_shards.h"

#include "realm/event.h"

#include "defer.h"
#include "reshard.h"
#include "reshard_manager.h"
#include "reshard_point_to_point.h"
#include "tiled_array.h"
#include "type_traits.h"

namespace zuku {
namespace {

void ScatterShardsToLocalTarget(const Processor& p, const ShardedArray& src,
                                Store<ShardedArray>& dst) {
  const int64_t num_shards = src.shape().sharding.devices.size();

  const int64_t dst_shard_id =
      *dst->shape().sharding.devices.ShardId(p.global_id());

  const int64_t shards_per_source = dst->shape().sharding.devices.size() /
                                    src.shape().sharding.devices.size();

  const int64_t src_shard_id = dst_shard_id / shards_per_source;

  ShardedArray::ReshardKey key{.src_id = src.unique_id(),
                               .dst_id = dst->unique_id(),
                               .src_shard_id = src_shard_id,
                               .dst_shard_id = dst_shard_id};
  const Realm::Event copy = ReshardManager::GetManager().ReshardToLocalTarget(
      p, dst->tile().realm_shape(), key, src, dst);

  if (log_reshard.want_debug()) {
    std::visit(
        [&](const auto& rect) {
          log_reshard.debug()
              << "copy to " << p.global_id() << " from shards " << src_shard_id
              << "->" << dst_shard_id << " for rectangle " << rect;
        },
        dst->tile().realm_shape());
  }

  dst.ReadyAfter(copy);
}

void ScatterShardsFromLocalSource(const Processor& p, View<ShardedArray>& src,
                                  const ShardedArray& dst) {
  const int64_t shards_per_source = dst.shape().sharding.devices.size() /
                                    src->shape().sharding.devices.size();
  const int64_t src_shard_id =
      *src->shape().sharding.devices.ShardId(p.global_id());

  const int64_t shard_offset = shards_per_source * src_shard_id;
  std::set<Realm::Event> all_events;
  for (int64_t shard = 0; shard < shards_per_source; ++shard) {
    const int64_t dst_shard_id = shard_offset + shard;
    ShardedArray::ReshardKey key{.src_id = src->unique_id(),
                                 .dst_id = dst.unique_id(),
                                 .src_shard_id = src_shard_id,
                                 .dst_shard_id = dst_shard_id};
    RealmShape bounds = ComputeTileBounds(dst_shard_id, dst.shape().sharding);
    const Realm::Event copy =
        ReshardManager::GetManager().ReshardFromLocalSource(p, bounds, key, src,
                                                            dst);
    all_events.insert(copy);
    if (log_reshard.want_debug()) {
      std::visit(
          [&](const auto& rect) {
            log_reshard.debug() << "copy from " << p.global_id()
                                << " from shards " << src_shard_id << "->"
                                << dst_shard_id << " for rectangle " << rect;
          },
          bounds);
    }
  }
  src.DoneAfter(Realm::Event::merge_events(all_events));
}

}  // namespace

ScatterShardsConfig GetScatterShardsConfig(const Processor& p,
                                           const Sharding& src,
                                           const Sharding& dst) {
  return ScatterShardsConfig{.dst_shard = dst.devices.ShardId(p.global_id()),
                             .sender = src.devices.Contains(p.global_id())};
}

void ReshardScatterShards(const Processor& p, const ScatterShardsConfig& config,
                          View<ShardedArray> src, Store<ShardedArray>& dst) {
  if (src->HasTile()) {
    ScatterShardsFromLocalSource(p, src, *dst);
  }

  if (dst->HasTile()) {
    ScatterShardsToLocalTarget(p, *src, dst);
  }
}

}  // namespace zuku
