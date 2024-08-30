/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "reshard_scatter_gather.h"

#include "realm/event.h"

#include "reshard.h"
#include "reshard_manager.h"
#include "tiled_array.h"
#include "type_traits.h"

namespace zuku {

std::ostream& operator<<(std::ostream& os, const SliceConfig& config) {
  os << "SliceConfig { .src_shard = " << config.src_shard
     << ", .dst_shard = " << config.dst_shard << ", .bounds = " << config.bounds
     << " }";
  return os;
}

std::ostream& operator<<(std::ostream& os, const ScatterGatherConfig& config) {
  os << "ScatterGatherConfig { .from_source = " << config.from_source
     << ", .to_target = " << config.to_target << " }";
  return os;
}

void ReshardScatterGatherFromLocalSource(const Processor& p,
                                         const ScatterGatherConfig& config,
                                         View<ShardedArray> src,
                                         const ShardedArray& dst) {
  ReshardManager& manager = ReshardManager::GetManager();
  std::set<Realm::Event> slice_events;
  for (const SliceConfig& slice : config.from_source) {
    ShardedArray::ReshardKey key{
        .src_id = src->unique_id(),
        .dst_id = dst.unique_id(),
        .src_shard_id = slice.src_shard,
        .dst_shard_id = slice.dst_shard,
    };
    if (log_reshard.want_debug()) {
      std::visit(overloaded{[&](auto&& bounds) {
                   log_reshard.debug()
                       << "resharding from source slice " << bounds
                       << " for shards " << slice.src_shard << "->"
                       << slice.dst_shard << " from instance "
                       << src->tile().instance();
                 }},
                 slice.bounds);
    }
    slice_events.insert(
        manager.ReshardFromLocalSource(p, slice.bounds, key, src, dst));
  }
  Realm::Event postcondition = Realm::Event::merge_events(slice_events);
  src.DoneAfter(postcondition);
}

void ReshardScatterGatherToLocalTarget(const Processor& p,
                                       const ScatterGatherConfig& config,
                                       const ShardedArray& src,
                                       Store<ShardedArray>& dst) {
  ReshardManager& manager = ReshardManager::GetManager();
  std::set<Realm::Event> slice_events;
  for (const SliceConfig& slice : config.to_target) {
    ShardedArray::ReshardKey key{
        .src_id = src.unique_id(),
        .dst_id = dst->unique_id(),
        .src_shard_id = slice.src_shard,
        .dst_shard_id = slice.dst_shard,
    };
    if (log_reshard.want_debug()) {
      std::visit(overloaded{[&](auto&& bounds) {
                   log_reshard.debug()
                       << "resharding to target slice " << bounds
                       << " for shards " << slice.src_shard << "->"
                       << slice.dst_shard << " to instance "
                       << dst->tile().instance();
                 }},
                 slice.bounds);
    }
    slice_events.insert(
        manager.ReshardToLocalTarget(p, slice.bounds, key, src, dst));
  }
  Realm::Event postcondition = Realm::Event::merge_events(slice_events);
  dst.ReadyAfter(postcondition);
}

void ReshardScatterGather(const Processor& p, const ScatterGatherConfig& config,
                          View<ShardedArray> src, Store<ShardedArray>& dst) {
  if (src->HasTile()) {
    ReshardScatterGatherFromLocalSource(p, config, src.view(), *dst);
  }

  if (dst->HasTile()) {
    ReshardScatterGatherToLocalTarget(p, config, *src, dst);
  }
}

}  // namespace zuku
