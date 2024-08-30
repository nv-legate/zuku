/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "reshard_broadcast.h"

#include "realm/event.h"

#include "defer.h"
#include "reshard.h"
#include "reshard_manager.h"
#include "reshard_point_to_point.h"
#include "tiled_array.h"
#include "type_traits.h"

namespace zuku {
namespace {

TargetConfig GetTargetConfig(int64_t src_shard, int64_t dest_root_shard,
                             const DeviceList& dst_devices, int64_t my_rank) {
  const int64_t num_leaves = dst_devices.size();
  if (my_rank % 2) {
    // must be an odd rank, just receive from -1
    return TargetConfig{.dst_root_shard = dest_root_shard + my_rank - 1};
  }

  // 0 -> N/2, N/4, N/8, ...
  // N/2 -> N/2 + N/4, N/2 + N/8
  // N/4 -> N/4 + N/8
  const int64_t next_power_of_2 = [&] {
    int64_t power = 1;
    while (power < num_leaves) {
      power *= 2;
    }
    return power;
  }();

  int64_t split_start = 0;
  int64_t split_size = next_power_of_2;
  while (split_start != my_rank) {
    split_size /= 2;
    if (my_rank > split_start) {
      split_start += split_size;
    } else if (my_rank < split_start) {
      split_start -= split_size;
    }
  }

  const int64_t my_source = dest_root_shard + my_rank - split_size;

  std::vector<int64_t> targets;
  split_size /= 2;
  int64_t next_leaf = split_start + split_size;
  while (next_leaf < num_leaves && split_size > 0) {
    targets.push_back(dest_root_shard + next_leaf);
    split_size /= 2;
    next_leaf -= split_size;
  }

  if (my_rank > 0) {
    return TargetConfig{
        .dst_root_shard = my_source,
        .dst_leaf_shards = std::move(targets),
    };
  }

  return TargetConfig{.src_shard = src_shard,
                      .dst_leaf_shards = std::move(targets)};
}

}  // namespace

BroadcastConfig GetBroadcastConfig(std::optional<int64_t> src_shard,
                                   std::optional<int64_t> dst_shard,
                                   const Sharding& src, const Sharding& dst) {
  const int64_t leaves_per_root =
      (dst.devices.size() + src.devices.size() - 1) / src.devices.size();

  std::optional<SourceConfig> source;
  if (src_shard.has_value()) {
    // Send your shard to a destination shard
    // Then have that destination shard act as the root of
    // an MPI-style bcast tree
    const int64_t dst_partner_shard = *src_shard * leaves_per_root;
    if (dst_partner_shard >= dst.devices.size()) {
      // this is an extra shard and does not participate
      return BroadcastConfig{};
    }
    source.emplace();
    source->dst_root_shards.push_back(dst_partner_shard);
  }

  std::optional<TargetConfig> target;
  if (dst_shard.has_value()) {
    const int64_t my_src_shard = *dst_shard / leaves_per_root;
    const int64_t my_rank = *dst_shard - my_src_shard * leaves_per_root;
    const int64_t num_leaves = std::min(
        leaves_per_root, dst.devices.size() - my_src_shard * leaves_per_root);
    const int64_t tree_offset = my_src_shard * leaves_per_root;
    target =
        GetTargetConfig(my_src_shard, tree_offset,
                        dst.devices.slice(tree_offset, num_leaves), my_rank);
  }

  return BroadcastConfig{.source = std::move(source),
                         .target = std::move(target)};
}

void ReshardBroadcastFromLocalSource(const Processor& p,
                                     const BroadcastConfig& config,
                                     View<ShardedArray> src,
                                     const ShardedArray& dst) {
  std::set<Realm::Event> sends;
  for (int64_t dst_shard : config.source->dst_root_shards) {
    ShardedArray::ReshardKey key{.src_id = src->unique_id(),
                                 .dst_id = dst.unique_id(),
                                 .src_shard_id = src->shard_id(),
                                 .dst_shard_id = dst_shard};
    log_reshard.debug() << "starting bcast tree from source shard "
                        << src->shard_id() << " on dest shard " << dst_shard
                        << " on " << p.global_id();
    sends.insert(ReshardManager::GetManager().ReshardFromLocalSource(
        p, src->tile().realm_shape(), key, src, dst));
  }
  src.DoneAfter(Realm::Event::merge_events(sends));
}

void ReshardBroadcastToLocalTarget(const Processor& p,
                                   const BroadcastConfig& config,
                                   const ShardedArray& src,
                                   Store<ShardedArray>& dst) {
  const Realm::Event initial_precondition = [&] {
    if (config.target->src_shard.has_value()) {
      ShardedArray::ReshardKey key{.src_id = src.unique_id(),
                                   .dst_id = dst->unique_id(),
                                   .src_shard_id = *config.target->src_shard,
                                   .dst_shard_id = dst->shard_id()};
      const Realm::Event copy =
          ReshardManager::GetManager().ReshardToLocalTarget(
              p, dst->tile().realm_shape(), key, src, dst);
      // we need to receive from the src before we can send to any leaeves
      log_reshard.debug() << "target starting bcast tree from source shard "
                          << *config.target->src_shard << " on "
                          << p.global_id() << " after " << copy;
      return copy;
    } else if (config.target->dst_root_shard.has_value()) {
      // we instead receive from another shard in the dst tensor
      ShardedArray::ReshardKey key{
          .src_id = dst->unique_id(),
          .dst_id = dst->unique_id(),
          .src_shard_id = *config.target->dst_root_shard,
          .dst_shard_id = dst->shard_id()};
      const Realm::Event copy =
          ReshardManager::GetManager().ReshardToLocalTarget(
              p, dst->tile().realm_shape(), key, *dst, dst);
      log_reshard.debug() << "starting bcast tree from dest shard "
                          << *config.target->dst_root_shard << " on "
                          << p.global_id() << " after " << copy;
      return copy;
    }
    return Realm::Event::NO_EVENT;
  }();

  dst.ReadyAfter(initial_precondition);

  if (!config.target->dst_leaf_shards.empty()) {
    View<ShardedArray> bcast_view = dst.view();
    for (int64_t shard : config.target->dst_leaf_shards) {
      ShardedArray::ReshardKey key{.src_id = dst->unique_id(),
                                   .dst_id = dst->unique_id(),
                                   .src_shard_id = dst->shard_id(),
                                   .dst_shard_id = shard};
      const Realm::Event copy =
          ReshardManager::GetManager().ReshardFromLocalSource(
              p, dst->tile().realm_shape(), key, bcast_view, *dst);
      bcast_view.DoneAfter(copy);
      log_reshard.debug() << "sending down bcast tree to shard " << shard
                          << " on " << p.global_id() << " after " << copy;
    }
  }
}  // namespace zuku

void ReshardBroadcast(const Processor& p, const BroadcastConfig& config,
                      View<ShardedArray> src, Store<ShardedArray>& dst) {
  if (dst->HasTile()) {
    ReshardBroadcastToLocalTarget(p, config, *src, dst);
  }

  if (src->HasTile()) {
    ReshardBroadcastFromLocalSource(p, config, std::move(src), *dst);
  }
}

}  // namespace zuku
