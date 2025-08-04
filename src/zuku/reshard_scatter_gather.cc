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
namespace {

template <int N, typename T>
RealmShape MakeTileBoxN(const std::vector<int64_t>& index,
                        const Sharding& sharding) {
  Realm::Rect<N, T> bounds;
  int64_t box_dim = 0;
  for (int64_t s = 0; s < sharding.dims.size(); ++s) {
    const int64_t tile_size = sharding.dims[s].size / sharding.dims[s].sharding;
    // skip replicated dimension
    if (tile_size > 0) {
      bounds.lo[box_dim] = index[box_dim] * tile_size;
      bounds.hi[box_dim] = bounds.lo[box_dim] + tile_size - 1;
      ++box_dim;
    }
  }
  return bounds;
}

}  // namespace

RealmShape MakeTileBox(const std::vector<int64_t>& index,
                       const Sharding& sharding) {
  const int64_t num_box_dims = [&] {
    if (sharding.IsPartiallyReplicated()) {
      return sharding.dims.size() - 1;
    }
    return sharding.dims.size();
  }();
  switch (num_box_dims) {
    case 0:
      return MakeTileBoxN<1, long long>(index, sharding);
    case 1:
      return MakeTileBoxN<1, long long>(index, sharding);
    case 2:
      return MakeTileBoxN<2, long long>(index, sharding);
    case 3:
      return MakeTileBoxN<3, long long>(index, sharding);
    case 4:
      return MakeTileBoxN<4, long long>(index, sharding);
#if REALM_MAX_DIM >= 5
    case 5:
      return MakeTileBoxN<5, long long>(index, sharding);
#endif
    default:
      throw std::runtime_error("unsupported no. dims");
  }
}

ScatterGatherConfig GetScatterGatherConfig(int64_t dim,
                                           std::optional<int64_t> src_shard,
                                           std::optional<int64_t> dst_shard,
                                           const Sharding& src,
                                           const Sharding& dst) {
  const int64_t src_sharding = src.dims[dim].sharding;
  const int64_t dst_sharding = dst.dims[dim].sharding;

  ScatterGatherConfig sg_config;

  assert(src_sharding != dst_sharding);
  if (src_sharding > dst_sharding) {
    const int64_t num_slices_per_dest =
        src.dims[dim].sharding / dst.dims[dim].sharding;
    const int64_t slice_size = src.dims[dim].size / src.dims[dim].sharding;
    if (src_shard.has_value()) {
      std::vector<int64_t> index = ComputeTileIndex(*src_shard, src);
      auto bounds = MakeTileBox(index, src);
      const int64_t offset = *src_shard % num_slices_per_dest;
      index[dim] /= num_slices_per_dest;
      const int64_t dst_shard = ComputeTileShard(index, dst);
      SliceConfig config{.src_shard = *src_shard,
                         .dst_shard = dst_shard,
                         .bounds = std::move(bounds)};
      sg_config.from_source.push_back(std::move(config));
    }

    if (dst_shard.has_value()) {
      // we are going to receive from multiple sources
      sg_config.to_target.reserve(num_slices_per_dest);
      std::vector<int64_t> index = ComputeTileIndex(*dst_shard, dst);
      const int64_t src_shard_start = index[dim] * num_slices_per_dest;
      for (int64_t slice = 0; slice < num_slices_per_dest; ++slice) {
        index[dim] = src_shard_start + slice;
        auto bounds = MakeTileBox(index, src);
        SliceConfig config{
            .src_shard = ComputeTileShard(index, src),
            .dst_shard = *dst_shard,
            .bounds = std::move(bounds),
        };
        sg_config.to_target.push_back(std::move(config));
      }
    }
  } else {  // dst_sharding > src_sharding
    const int64_t num_slices_per_source =
        dst.dims[dim].sharding / src.dims[dim].sharding;
    const int64_t slice_size = dst.dims[dim].size / dst.dims[dim].sharding;
    if (dst_shard.has_value()) {
      std::vector<int64_t> index = ComputeTileIndex(*dst_shard, dst);
      auto bounds = MakeTileBox(index, dst);
      index[dim] /= num_slices_per_source;
      const int64_t src_shard = ComputeTileShard(index, src);
      SliceConfig config{
          .src_shard = src_shard,
          .dst_shard = *dst_shard,
          .bounds = std::move(bounds),
      };
      sg_config.to_target.push_back(std::move(config));
    }

    if (src_shard.has_value()) {
      sg_config.from_source.reserve(num_slices_per_source);

      std::vector<int64_t> index = ComputeTileIndex(*src_shard, src);
      const int64_t dest_idx_start = index[dim] * num_slices_per_source;
      for (int64_t slice = 0; slice < num_slices_per_source; ++slice) {
        index[dim] = dest_idx_start + slice;
        auto bounds = MakeTileBox(index, dst);
        SliceConfig config{
            .src_shard = *src_shard,
            .dst_shard = ComputeTileShard(index, dst),
            .bounds = std::move(bounds),
        };
        sg_config.from_source.push_back(std::move(config));
      }
    }
  }
  return sg_config;
}

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
