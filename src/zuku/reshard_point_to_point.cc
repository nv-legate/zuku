/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "reshard_point_to_point.h"

#include "realm/event.h"
#include "realm/logging.h"
#include "realm/processor.h"

#include <string>

#include "defer.h"
#include "future.h"
#include "init.h"
#include "processor.h"
#include "reshard.h"
#include "reshard_manager.h"
#include "tiled_array.h"

namespace zuku {

Realm::Event ReshardFromLocalSource(Processor p, const RealmShape& shape,
                                    View<ShardedArray> src,
                                    const ShardedArray& dst,
                                    int64_t src_shard_id,
                                    int64_t dst_shard_id) {
  ShardedArray::ReshardKey key{
      .src_id = src->unique_id(),
      .dst_id = dst.unique_id(),
      .src_shard_id = src_shard_id,
      .dst_shard_id = dst_shard_id,
  };
  ReshardManager& manager = ReshardManager::GetManager();
  const Realm::Event postcondition =
      manager.ReshardFromLocalSource(p, shape, key, src, dst);
  src.DoneAfter(postcondition);
  log_reshard.debug() << src->tile().instance()
                      << " will give up reference after " << postcondition;
  return postcondition;
}

Realm::Event ReshardToLocalTarget(Processor p, const RealmShape& shape,
                                  const ShardedArray& src,
                                  Store<ShardedArray>& dst,
                                  int64_t src_shard_id, int64_t dst_shard_id) {
  Realm::Barrier prebar, postbar;
  ShardedArray::ReshardKey key{
      .src_id = src.unique_id(),
      .dst_id = dst->unique_id(),
      .src_shard_id = src_shard_id,
      .dst_shard_id = dst_shard_id,
  };
  ReshardManager& manager = ReshardManager::GetManager();
  const Realm::Event postcondition =
      manager.ReshardToLocalTarget(p, shape, key, src, dst);
  // The postbar becomes the destination ready after event
  dst.ReadyAfter(postcondition);
  return postcondition;
}

void ReshardToFrom(Processor p, View<ShardedArray> src,
                   Store<ShardedArray> dst) {
  throw std::runtime_error("do not currently support non-disjoint shardings");
}

void ReshardPointToPoint(Processor p, zuku::View<zuku::ShardedArray> src,
                         zuku::Store<zuku::ShardedArray>& dst) {
  if (dst->HasTile()) {
    RealmShape shape = dst->tile().realm_shape();
    Realm::Event postcondition = ReshardToLocalTarget(
        p, shape, *src, dst, dst->shard_id(), dst->shard_id());
    log_reshard.debug() << "on proc " << p.global_id()
                        << " reshard recv finishing after " << postcondition;
  }
  if (src->HasTile()) {
    const int64_t src_unique_id = src->mesh_unique_id();
    const int64_t dst_unique_id = dst->mesh_unique_id();
    const int64_t shard_id = src->shard_id();
    RealmShape shape = src->tile().realm_shape();
    Realm::Event postcondition = ReshardFromLocalSource(
        p, shape, std::move(src), *dst, shard_id, shard_id);
    log_reshard.debug() << "on proc " << p.global_id()
                        << " reshard send finishing after " << postcondition;
  }
}

}  // namespace zuku
