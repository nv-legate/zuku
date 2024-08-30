/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef _POC_SRC_RESHARD_MANAGER_H_
#define _POC_SRC_RESHARD_MANAGER_H_

#include "realm/event.h"
#include "realm/processor.h"

#include <queue>

#include "reshard_scatter_gather.h"
#include "store.h"
#include "tiled_array.h"

namespace zuku {

class ReshardManager {
 private:
  // Make sure there's only one instance of the reshard manager
  ReshardManager(void) {}
  ~ReshardManager(void);

 public:
  // Overloads for point-to-point reshard
  Realm::Event ReshardFromLocalSource(const Processor& p,
                                      const RealmShape& shape,
                                      const ShardedArray::ReshardKey& key,
                                      const View<ShardedArray>& src,
                                      const ShardedArray& dst);
  Realm::Event ReshardToLocalTarget(const Processor& p, const RealmShape& shape,
                                    const ShardedArray::ReshardKey& key,
                                    const ShardedArray& src,
                                    const Store<ShardedArray>& dst);
  void ReceivePointToPointNames(const ShardedArray::ReshardKey& key,
                                Realm::Barrier barrier,
                                Realm::RegionInstance instance);

 public:
  // Overloads for scatter-gather reshard
  Realm::Event ReshardFromLocalSource(const Processor& p,
                                      const ScatterGatherConfig& config,
                                      const View<ShardedArray>& src,
                                      const ShardedArray& dst);
  Realm::Event ReshardToLocalTarget(const Processor& p,
                                    const ScatterGatherConfig& config,
                                    const ShardedArray& src,
                                    const Store<ShardedArray>& dst);
  void ReceiveScatterGatherNames(const ShardedArray::ScatterGatherKey& key,
                                 int64_t dst_shard, Realm::RegionInstance dst,
                                 Realm::Event precondition,
                                 Realm::UserEvent postcondition);

 public:
  // We'll have a singleton reshard manager that we'll always be able to lookup
  static ReshardManager& GetManager(void);

  // Realm message function for pointwise name exchange
  static void PointToPointNameHandler(const void* args, size_t arglen,
                                      const void* userdata, size_t usersize,
                                      Realm::Processor processor);

 private:
  Realm::FastReservation mutex_;

  Realm::Event RendezvousReshardFromLocalSource(
      const Processor& p, const RealmShape& shape,
      const ShardedArray::ReshardKey& key, const View<ShardedArray>& src,
      const ShardedArray& dst);

  Realm::Event EagerReshardFromLocalSource(const Processor& p,
                                           const RealmShape& shape,
                                           const ShardedArray::ReshardKey& key,
                                           const View<ShardedArray>& src,
                                           const ShardedArray& dst);

  Realm::Event RendezvousReshardToLocalTarget(
      const Processor& p, const RealmShape& shape,
      const ShardedArray::ReshardKey& key, const ShardedArray& src,
      const Store<ShardedArray>& dst);

  Realm::Event EagerReshardToLocalTarget(const Processor& p,
                                         const RealmShape& shape,
                                         const ShardedArray::ReshardKey& key,
                                         const ShardedArray& src,
                                         const Store<ShardedArray>& dst);

  struct PointToPointReshard {
    struct PendingOp {
      Realm::Event precondition;
      Realm::UserEvent postcondition_to_trigger{
          Realm::UserEvent::NO_USER_EVENT};
      Realm::RegionInstance partner_instance{Realm::RegionInstance::NO_INST};
    };
    Realm::Barrier target_ready_barrier{Realm::Barrier::NO_BARRIER};
    Realm::Barrier ack_done_barrier{Realm::Barrier::NO_BARRIER};
    RealmShape shape;
    size_t field_size{0};
    std::queue<PendingOp> pending_source_reshards;
    std::queue<PendingOp> pending_dest_reshards;
    bool exists{false};
    std::optional<std::string> name;
  };
  // Make sure to index this map using the global ids and not the Realm
  // instances because Realm instances can be collected and IDs recycled. We'll
  // presume that the JAX global IDs are unique for all time.
  std::unordered_map<ShardedArray::ReshardKey, PointToPointReshard>
      point_to_point_reshards_;

  struct PointToPointNameArgs {
   public:
    PointToPointNameArgs(const ShardedArray::ReshardKey& k, Realm::Barrier b,
                         Realm::RegionInstance i)
        : key(k), barrier(b), instance(i) {}

   public:
    ShardedArray::ReshardKey key;
    Realm::Barrier barrier;
    Realm::RegionInstance instance;
  };
  static_assert(std::is_trivially_copyable<PointToPointNameArgs>::value);

 private:
  Realm::Event RemoteCopyNextPendingToTarget(PointToPointReshard& reshard,
                                             Realm::RegionInstance src_instance,
                                             Realm::RegionInstance dst_instance,
                                             Realm::Event ready_barrier,
                                             Realm::Event src_precondition);
};

}  // namespace zuku

#endif
