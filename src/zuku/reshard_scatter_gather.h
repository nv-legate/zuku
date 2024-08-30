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

template <int N, typename T>
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
      Realm::Rect<N, T> bounds;
      for (int64_t i = 0; i < N; ++i) {
        const int64_t tile_size = src.dims[i].size / src.dims[i].sharding;
        bounds.lo[i] = index[i] * tile_size;
        bounds.hi[i] = bounds.lo[i] + tile_size - 1;
      }

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
        Realm::Rect<N, T> bounds;
        for (int64_t i = 0; i < N; ++i) {
          const int64_t tile_size = src.dims[i].size / src.dims[i].sharding;
          bounds.lo[i] = index[i] * tile_size;
          bounds.hi[i] = bounds.lo[i] + tile_size - 1;
        }
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
      Realm::Rect<N, T> bounds;
      for (int64_t i = 0; i < N; ++i) {
        const int64_t tile_size = dst.dims[i].size / dst.dims[i].sharding;
        bounds.lo[i] = index[i] * tile_size;
        bounds.hi[i] = bounds.lo[i] + tile_size - 1;
      }
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
        Realm::Rect<N, T> bounds;
        for (int64_t i = 0; i < N; ++i) {
          const int64_t tile_size = dst.dims[i].size / dst.dims[i].sharding;
          bounds.lo[i] = index[i] * tile_size;
          bounds.hi[i] = bounds.lo[i] + tile_size - 1;
        }
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

void ReshardScatterGather(const Processor& p, const ScatterGatherConfig& config,
                          View<zuku::ShardedArray> src,
                          Store<zuku::ShardedArray>& dst);

}  // namespace zuku

#endif
