/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "reshard_gather_shards.h"

#include "realm/event.h"

#include "reshard.h"
#include "reshard_manager.h"
#include "tiled_array.h"

namespace zuku {
namespace {

void GatherShardsToLocalTarget(const Processor& p, const ShardedArray& src,
                               Store<ShardedArray>& dst,
                               const GatherShardsConfig& config) {
  const int64_t num_shards = src.shape().sharding.devices.size();
  const int64_t dst_shard_id = *config.dst_shard;

  std::set<Realm::Event> all_events;
  for (int64_t shard = 0; shard < num_shards; ++shard) {
    ShardedArray::ReshardKey key{.src_id = src.unique_id(),
                                 .dst_id = dst->unique_id(),
                                 .src_shard_id = shard,
                                 .dst_shard_id = dst_shard_id};

    const Realm::Event copy = ReshardManager::GetManager().ReshardToLocalTarget(
        p, config.target_from_source_bounds[shard], key, src, dst);
    all_events.insert(copy);

    log_reshard.debug() << "receiving local shard " << shard << " "
                        << config.target_from_source_bounds[shard] << " on "
                        << p.global_id() << " for " << src.shape().sharding
                        << "->" << dst->shape().sharding;
  }
  dst.ReadyAfter(Realm::Event::merge_events(all_events));
}

void GatherShardsFromLocalSource(const Processor& p, View<ShardedArray>& src,
                                 const ShardedArray& dst,
                                 const GatherShardsConfig& config) {
  const int64_t num_shards = dst.shape().sharding.devices.size();
  const int64_t src_shard_id =
      *src->shape().sharding.devices.ShardId(p.global_id());
  std::set<Realm::Event> all_events;

  for (int64_t shard = 0; shard < num_shards; ++shard) {
    ShardedArray::ReshardKey key{.src_id = src->unique_id(),
                                 .dst_id = dst.unique_id(),
                                 .src_shard_id = src_shard_id,
                                 .dst_shard_id = shard};
    const Realm::Event copy =
        ReshardManager::GetManager().ReshardFromLocalSource(
            p, *config.source_to_target_bounds, key, src, dst);
    all_events.insert(copy);
    log_reshard.debug() << "sending local shard " << src_shard_id << " "
                        << *config.source_to_target_bounds << " on "
                        << p.global_id() << " for " << src->shape().sharding
                        << "->" << dst.shape().sharding;
  }
  src.DoneAfter(Realm::Event::merge_events(all_events));
}

}  // namespace

std::ostream& operator<<(std::ostream& os, const GatherShardsConfig& config) {
  os << "GatherShardsConfig{ .src_shard = " << config.src_shard
     << ", .dst_shard = " << config.dst_shard
     << ", .source_to_target_bounds = " << config.source_to_target_bounds
     << ", .target_from_source_bounds = " << config.target_from_source_bounds
     << " }";
  return os;
}

bool operator==(const GatherShardsConfig& lhs, const GatherShardsConfig& rhs) {
  return lhs.src_shard == rhs.src_shard && lhs.dst_shard == rhs.dst_shard &&
         lhs.source_to_target_bounds == rhs.source_to_target_bounds &&
         lhs.target_from_source_bounds == rhs.target_from_source_bounds;
}

GatherShardsConfig GetGatherShardsConfig(const Processor& p,
                                         const Sharding& src,
                                         const Sharding& dst) {
  auto src_shard = src.devices.ShardId(p.global_id());
  auto dst_shard = dst.devices.ShardId(p.global_id());
  auto dest_bounds = [&]() -> std::optional<RealmShape> {
    if (src_shard.has_value()) {
      return ComputeTileBounds(*src_shard, src);
    }
    return std::nullopt;
  }();

  const int64_t num_src_shards = src.devices.size();
  std::vector<RealmShape> source_bounds;
  source_bounds.reserve(num_src_shards);
  for (int64_t shard = 0; shard < num_src_shards; ++shard) {
    RealmShape bounds = ComputeTileBounds(shard, src);
    source_bounds.push_back(std::move(bounds));
  }

  return {.src_shard = std::move(src_shard),
          .dst_shard = std::move(dst_shard),
          .source_to_target_bounds = std::move(dest_bounds),
          .target_from_source_bounds = std::move(source_bounds)};
}

void ReshardGatherShards(const Processor& p, const GatherShardsConfig& config,
                         View<ShardedArray> src, Store<ShardedArray>& dst) {
  if (src->HasTile()) {
    if (!config.src_shard.has_value()) {
      std::stringstream sstr;
      sstr << "ReshardGatherShards: unexpected mismatch between source tile "
              "and config on "
           << p << " for " << src->shape() << " -> " << dst->shape();
      throw std::runtime_error(sstr.str());
    }
    GatherShardsFromLocalSource(p, src, *dst, config);
  }

  if (dst->HasTile()) {
    if (!config.dst_shard.has_value()) {
      std::stringstream sstr;
      sstr << "ReshardGatherShards: unexpected mismatch between destination "
              "tile and config on "
           << p << " for " << src->shape() << " -> " << dst->shape();
      throw std::runtime_error(sstr.str());
    }
    GatherShardsToLocalTarget(p, *src, dst, config);
  }
}

}  // namespace zuku
