/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef _POC_SRC_RESHARD_BROADCAST_H_
#define _POC_SRC_RESHARD_BROADCAST_H_

#include "realm.h"
#include "store.h"
#include "tiled_array.h"
#include "zuku_utils.h"

namespace zuku {

struct SourceConfig {
  std::vector<int64_t> dst_root_shards;
};
inline bool operator==(const SourceConfig& lhs, const SourceConfig& rhs) {
  return lhs.dst_root_shards == rhs.dst_root_shards;
}

inline std::ostream& operator<<(std::ostream& os, const SourceConfig& config) {
  os << "SourceConfig{ .dst_root_shards = " << config.dst_root_shards << " }";
  return os;
}

struct TargetConfig {
  // if this receives its data from the source tensor
  std::optional<int64_t> src_shard;
  // if this receives its data from a destination shard
  std::optional<int64_t> dst_root_shard;
  // if this forwards its data to other destination shards
  std::vector<int64_t> dst_leaf_shards;
};
inline bool operator==(const TargetConfig& lhs, const TargetConfig& rhs) {
  return lhs.src_shard == rhs.src_shard &&
         lhs.dst_root_shard == rhs.dst_root_shard &&
         lhs.dst_leaf_shards == rhs.dst_leaf_shards;
}

inline std::ostream& operator<<(std::ostream& os, const TargetConfig& config) {
  os << "TargetConfig{ .src_shard = " << config.src_shard
     << ", .dst_root_shard = " << config.dst_root_shard
     << ", .dst_leaf_shards = " << config.dst_leaf_shards << " }";
  return os;
}

struct BroadcastConfig {
  // exists if this is a leaf that needs
  // to receive from a root
  std::optional<SourceConfig> source;
  std::optional<TargetConfig> target;
};

BroadcastConfig GetBroadcastConfig(std::optional<int64_t> src_shard,
                                   std::optional<int64_t> dst_shard,
                                   const Sharding& src, const Sharding& dst);

void ReshardBroadcast(const Processor& p, const BroadcastConfig& config,
                      View<ShardedArray> src, Store<ShardedArray>& dst);

inline bool operator==(const BroadcastConfig& lhs, const BroadcastConfig& rhs) {
  return lhs.source == rhs.source && lhs.target == rhs.target;
}

inline std::ostream& operator<<(std::ostream& os,
                                const BroadcastConfig& config) {
  os << "BroadcastConfig{ .source = " << config.source
     << ", .target = " << config.target << " }";
  return os;
}

}  // namespace zuku

#endif
