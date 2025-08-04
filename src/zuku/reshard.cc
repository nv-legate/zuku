/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "reshard.h"
#include <limits>

#include "realm/logging.h"
#include "reshard_gather_shards.h"
#include "reshard_point_to_point.h"

#include "reshard_scatter_gather.h"
#include "shape.h"

namespace zuku {
namespace {

template <int N, typename T>
Realm::Rect<N, T> SliceDim(const Realm::Rect<N, T>& rect, int64_t slice_number,
                           int64_t total_slices, int64_t dim) {
  int64_t extent = rect.hi[dim] - rect.lo[dim] + 1;
  int64_t slice_size = extent / total_slices;
  int64_t offset = slice_size * slice_number;
  Realm::Rect<N, T> new_rect = rect;
  new_rect.lo[dim] = rect.lo[dim] + offset;
  new_rect.hi[dim] = new_rect.lo[dim] + slice_size - 1;
  return new_rect;
}

RealmShape SliceDim(const RealmShape& shape, int64_t slice_number,
                    int64_t total_slices, int64_t dim) {
  return std::visit(
      overloaded{[&](const auto& rect) {
        return RealmShape(SliceDim(rect, slice_number, total_slices, dim));
      }},
      shape);
}

}  // namespace

Realm::Logger log_reshard("zuku-reshard");

ReshardConfigVariant DetermineShardingConfigReplicationChange(
    const Processor& p, const Sharding& src, const Sharding& dst) {
  return UnsupportedConfig{};
}

ReshardConfigVariant DetermineShardingConfigFullyReplicated(
    const Processor& p, const Sharding& src, const Sharding& dst) {
  if (src.devices.size() >= dst.devices.size()) {
    const int64_t src_shard_id = p.global_id() - src.devices.start();
    if (!dst.devices.Contains(p.global_id()) &&
        src_shard_id >= dst.devices.size()) {
      return NoOpConfig{};
    }

    // this is either sending or receivng a valid shard
    // and we can just pair up src and dst
    return PointToPointConfig{};
  }

  return GetBroadcastConfig(src.devices.ShardId(p.global_id()),
                            dst.devices.ShardId(p.global_id()), src, dst);
}

ReshardConfigVariant DetermineScatterGatherConfig(const Processor& p,
                                                  const Sharding& src,
                                                  const Sharding& dst) {
  int64_t num_dims_different{0};
  int64_t last_dim_different{-1};
  for (int64_t dim = 0; dim < src.dims.size(); ++dim) {
    if (src.dims[dim] != dst.dims[dim]) {
      num_dims_different++;
      last_dim_different = dim;
    }
  }

  if (num_dims_different == 1) {
    const int64_t src_sharding = src.dims[last_dim_different].sharding;
    const int64_t dst_sharding = dst.dims[last_dim_different].sharding;

    const auto src_shard = [&]() -> std::optional<int64_t> {
      if (src.devices.Contains(p.global_id())) {
        return p.global_id() - src.devices.start();
      }
      return std::nullopt;
    }();

    const auto dst_shard = [&]() -> std::optional<int64_t> {
      if (dst.devices.Contains(p.global_id())) {
        return p.global_id() - dst.devices.start();
      }
      return std::nullopt;
    }();

    return GetScatterGatherConfig(last_dim_different, src_shard, dst_shard, src,
                                  dst);
  }
  return UnsupportedConfig{};
}

ReshardConfigVariant DeterminePartialToReplicatedConfig(const Processor& p,
                                                        const Sharding& src,
                                                        const Sharding& dst) {
  const int64_t replicated_dim = [&] {
    int dim_number = 0;
    for (const auto& dim : src.dims) {
      if (dim.size == 1 && dim.sharding > 1) {
        return dim_number;
      }
      ++dim_number;
    }
    return -1;
  }();
  const int64_t replication = src.dims[replicated_dim].sharding;
  const int64_t repl_dim_permutation = src.dims[replicated_dim].permutation;
  std::optional<int64_t> slice_dim{std::nullopt};
  int64_t dim_number = 0;
  for (const auto& dim : src.dims) {
    if (dim.size == 1 && dim.sharding > 1) {
      continue;
    }
    const int64_t new_sharding = dim.sharding * replication;
    if (!slice_dim.has_value() && dim.size % new_sharding == 0) {
      slice_dim = dim_number;
    }
    ++dim_number;
  }
  if (!slice_dim.has_value()) {
    return UnsupportedConfig{};
  }

  auto src_shard = src.devices.ShardId(p.global_id());
  auto dst_shard = dst.devices.ShardId(p.global_id());
  auto dest_bounds = [&]() -> std::optional<RealmShape> {
    if (src_shard.has_value()) {
      std::vector<int64_t> index = ComputeTileIndex(*src_shard, src);
      const int64_t replica_number = index[replicated_dim];
      auto unsliced_bounds = ComputeTileBounds(*src_shard, src);
      return SliceDim(unsliced_bounds, replica_number, replication, *slice_dim);
    }
    return std::nullopt;
  }();

  const int64_t num_src_shards = src.devices.size();
  std::vector<RealmShape> source_bounds;
  source_bounds.reserve(num_src_shards);
  for (int64_t shard = 0; shard < num_src_shards; ++shard) {
    RealmShape unsliced_bounds = ComputeTileBounds(shard, src);
    std::vector<int64_t> index = ComputeTileIndex(shard, src);
    const int64_t replica_number = index[replicated_dim];
    source_bounds.push_back(
        SliceDim(unsliced_bounds, replica_number, replication, *slice_dim));
  }

  return GatherShardsConfig{
      .src_shard = std::move(src_shard),
      .dst_shard = std::move(dst_shard),
      .source_to_target_bounds = std::move(dest_bounds),
      .target_from_source_bounds = std::move(source_bounds)};
}

ReshardConfigVariant DetermineShardingConfig(const Processor& p,
                                             const Sharding& src,
                                             const Sharding& dst) {
  log_reshard.debug() << "determining sharding config for p=" << p.global_id()
                      << " " << src << "->" << dst;
  if (src.FullySharded() && dst.IsReplicated()) {
    return GetGatherShardsConfig(p, src, dst);
  }

  if (dst.FullySharded() && src.IsReplicated()) {
    return GetScatterShardsConfig(p, src, dst);
  }

  if (src.IsReplicated() && dst.IsReplicated()) {
    return DetermineShardingConfigFullyReplicated(p, src, dst);
  }

  if (src.IsPartiallyReplicated() && dst.IsReplicated()) {
    return DeterminePartialToReplicatedConfig(p, src, dst);
  }

  if (src.dims.size() != dst.dims.size()) {
    return DetermineShardingConfigReplicationChange(p, src, dst);
  }

  if (src.devices.size() == dst.devices.size() && src.dims == dst.dims) {
    return PointToPointConfig{};
  }

  return DetermineScatterGatherConfig(p, src, dst);
}  // namespace zuku

void Reshard(Processor p, zuku::View<ShardedArray> src,
             zuku::Store<ShardedArray>& dst) {
  auto reshard_config_variant =
      DetermineShardingConfig(p, src->shape().sharding, dst->shape().sharding);
  std::visit(overloaded{[&](const PointToPointConfig& cfg) {
                          ReshardPointToPoint(p, std::move(src), dst);
                        },
                        [&](const ScatterGatherConfig& cfg) {
                          ReshardScatterGather(p, cfg, std::move(src), dst);
                        },
                        [&](const BroadcastConfig& cfg) {
                          ReshardBroadcast(p, cfg, std::move(src), dst);
                        },
                        [&](const NoOpConfig& cfg) {},
                        [&](const ScatterShardsConfig& cfg) {
                          ReshardScatterShards(p, cfg, std::move(src), dst);
                        },
                        [&](const GatherShardsConfig& cfg) {
                          ReshardGatherShards(p, cfg, std::move(src), dst);
                        },
                        [&](auto cfg) {
                          log_reshard.warning()
                              << "UNSUPPORTED RESHARD:" << src->shape() << "->"
                              << dst->shape();
                          throw std::runtime_error(
                              "unsupported resharding pattern requested");
                        }}  // namespace zuku
             ,
             reshard_config_variant);
}  // namespace zuku

void Reshard(Processor p, zuku::Store<ShardedArray>& src,
             zuku::Store<ShardedArray>& dst) {
  Reshard(p, src.view(), dst);
}

}  // namespace zuku
