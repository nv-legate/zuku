/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "reshard_manager.h"

#include "realm/event.h"
#include "realm/logging.h"
#include "realm/processor.h"

#include <string>

#include "defer.h"
#include "init.h"
#include "processor.h"
#include "reshard.h"
#include "shape.h"
#include "tiled_array.h"
#include "type_traits.h"

namespace zuku {

namespace {

enum class Protocol { RENDEZVOUS, EAGER };
// TODO: enable eager by changing cutoff
constexpr size_t kEagerCutoffBytes = 0;

Protocol GetProtocol(const ArrayTile& tile) {
  if (tile.byte_size() < kEagerCutoffBytes) {
    return Protocol::EAGER;
  }
  return Protocol::RENDEZVOUS;
}

std::string ReshardName(const ShardedArray& src, const ShardedArray& dst) {
  const auto [shard_id, byte_size] = [&] {
    if (src.HasTile()) {
      return std::make_tuple(src.shard_id(), src.tile().byte_size());
    }
    return std::make_tuple(dst.shard_id(), dst.tile().byte_size());
  }();

  const int64_t src_device = src.shape().sharding.devices[shard_id];
  const int64_t dst_device = dst.shape().sharding.devices[shard_id];

  std::stringstream sstr;
  sstr << "reshard ";
  if (dst.HasName()) {
    sstr << src.name();
  }
  sstr << " [";
  size_t dim = 0;
  for (const ShardingDim& sdim : dst.shape().sharding.dims) {
    sstr << sdim.size;
    ++dim;
    if (dim < dst.shape().sharding.dims.size()) {
      sstr << ",";
    }
  }
  sstr << "]%[";
  dim = 0;
  for (const ShardingDim& sdim : dst.shape().sharding.dims) {
    sstr << sdim.sharding;
    ++dim;
    if (dim < dst.shape().sharding.dims.size()) {
      sstr << ",";
    }
  }
  sstr << "]: " << byte_size << " bytes " << src_device << "->" << dst_device;
  return sstr.str();
}

}  // namespace

/*static*/ ReshardManager& ReshardManager::GetManager() {
  static ReshardManager singleton;
  return singleton;
}

ReshardManager::~ReshardManager() {
  for (auto& [key, reshard] : point_to_point_reshards_) {
    if (reshard.ack_done_barrier.exists()) {
      reshard.ack_done_barrier.destroy_barrier();
    }
  }
}

Realm::Event IssueCopy(const RealmShape& shape, size_t field_size,
                       const Realm::RegionInstance& src_instance,
                       const Realm::RegionInstance& dst_instance,
                       Realm::Event precondition) {
  std::vector<Realm::CopySrcDstField> srcs(1);
  srcs[0].set_field(src_instance, 0 /*field id*/, field_size);

  std::vector<Realm::CopySrcDstField> dsts(1);
  dsts[0].set_field(dst_instance, 0 /*field id*/, field_size);

  const int priority = 0;

  return std::visit(overloaded{[&](auto&& bounds) {
                      const Realm::ProfilingRequestSet no_requests;
                      const Realm::Event postcondition = bounds.copy(
                          srcs, dsts, no_requests, precondition, priority);
                      log_reshard.debug()
                          << "issuing copy " << postcondition << " from "
                          << srcs[0].inst << "->" << dsts[0].inst
                          << " on bounds " << bounds << " after "
                          << precondition;
                      return postcondition;
                    }},
                    shape);
}

Realm::Event ReshardManager::RemoteCopyNextPendingToTarget(
    PointToPointReshard& reshard, Realm::RegionInstance src_instance,
    Realm::RegionInstance dst_instance, Realm::Event ready_barrier,
    Realm::Event src_precondition) {
  Realm::Event precondition =
      Realm::Event::merge_events(ready_barrier, src_precondition);
  Realm::Event postcondition =
      IssueCopy(reshard.shape, reshard.field_size, src_instance, dst_instance,
                precondition);
  log_reshard.debug() << "remote copy " << postcondition
                      << " after target ready barrier " << ready_barrier;
  // reshard.target_ready_barrier =
  // reshard.target_ready_barrier.advance_barrier();
  assert(reshard.ack_done_barrier.exists());
  log_reshard.debug() << "acking done on " << reshard.ack_done_barrier
                      << " after " << postcondition;
  reshard.ack_done_barrier.arrive(1, postcondition);
  reshard.ack_done_barrier = reshard.ack_done_barrier.advance_barrier();
  return postcondition;
}

void ReshardManager::ReceivePointToPointNames(
    const ShardedArray::ReshardKey& key, Realm::Barrier barrier,
    Realm::RegionInstance instance) {
  mutex_.wrlock().wait();
  PointToPointReshard& reshard = point_to_point_reshards_[key];
  if (instance.exists()) {
    log_reshard.debug() << "source received remote instance " << instance;
    assert(barrier.exists());
    if (!reshard.pending_source_reshards.empty()) {
      auto& op = reshard.pending_source_reshards.front();
      Realm::Event postcondition = RemoteCopyNextPendingToTarget(
          reshard, op.partner_instance, instance, barrier, op.precondition);
      op.postcondition_to_trigger.trigger(postcondition);
      reshard.pending_source_reshards.pop();
    } else {
      reshard.pending_dest_reshards.push(PointToPointReshard::PendingOp{
          .precondition = barrier,
          .postcondition_to_trigger = Realm::UserEvent::NO_USER_EVENT,
          .partner_instance = instance,
      });
    }
  } else {
    assert(barrier.exists());
    reshard.ack_done_barrier = barrier;
    log_reshard.debug() << "target received remote barrier " << barrier;
    while (!reshard.pending_dest_reshards.empty()) {
      auto& op = reshard.pending_dest_reshards.front();
      Realm::Event postcondition = reshard.ack_done_barrier;
      log_reshard.debug()
          << "clearing pending target reshard after receiving barrier "
          << reshard.ack_done_barrier << ", triggering "
          << op.postcondition_to_trigger;
      reshard.ack_done_barrier = reshard.ack_done_barrier.advance_barrier();
      assert(op.postcondition_to_trigger.exists());
      op.postcondition_to_trigger.trigger(postcondition);
      reshard.pending_dest_reshards.pop();
    }
  }
  mutex_.unlock();
}

/*static*/ void ReshardManager::PointToPointNameHandler(
    const void* args, size_t arglen, const void* userdata, size_t usersize,
    Realm::Processor processor) {
  assert(arglen == sizeof(PointToPointNameArgs));
  const PointToPointNameArgs* pargs =
      static_cast<const PointToPointNameArgs*>(args);
  ReshardManager& manager = GetManager();
  manager.ReceivePointToPointNames(pargs->key, pargs->barrier, pargs->instance);
}

Realm::Event ReshardManager::ReshardFromLocalSource(
    const Processor& p, const RealmShape& shape,
    const ShardedArray::ReshardKey& key, const View<ShardedArray>& src,
    const ShardedArray& dst) {
  switch (GetProtocol(src->tile())) {
    case Protocol::EAGER:
      return EagerReshardFromLocalSource(p, shape, key, src, dst);
    case Protocol::RENDEZVOUS:
      return RendezvousReshardFromLocalSource(p, shape, key, src, dst);
  }
  // make the compiler happy
  return Realm::Event::NO_EVENT;
}

Realm::Event ReshardManager::EagerReshardFromLocalSource(
    const Processor& p, const RealmShape& shape,
    const ShardedArray::ReshardKey& key, const View<ShardedArray>& src,
    const ShardedArray& dst) {
  throw std::runtime_error(
      "ReshardManager::EagerReshardFromLocalSource: not implemented");
}

Realm::Event ReshardManager::RendezvousReshardFromLocalSource(
    const Processor& p, const RealmShape& shape,
    const ShardedArray::ReshardKey& key, const View<ShardedArray>& src,
    const ShardedArray& dst) {
  log_reshard.debug() << "RDVZ: send " << src->shape() << " shards "
                      << key.src_shard_id << "-> " << key.dst_shard_id << " on "
                      << p.global_id();

  // Check in read only mode to see if we've got all the data needed for issuing
  // the copy
  {
    mutex_.wrlock().wait();
    PointToPointReshard& reshard = point_to_point_reshards_[key];
    if (!reshard.exists) {
      reshard.field_size = SupportedTypeSizeOf(src->shape().type);
      reshard.shape = shape;
      reshard.name = ReshardName(*src, dst);
      reshard.exists = true;
    }

    Realm::Event postcondition = [&] {
      Realm::Processor target = dst.HostForShard(key.dst_shard_id);
      if (p.RealmProc().address_space() != target.address_space() &&
          !reshard.ack_done_barrier.exists()) {
        reshard.ack_done_barrier =
            Realm::Barrier::create_barrier(1 /*arrival count*/);
        PointToPointNameArgs args(key, reshard.ack_done_barrier,
                                  Realm::RegionInstance::NO_INST);
        log_reshard.debug()
            << "sending ack barrier=" << reshard.ack_done_barrier
            << " to reshard target " << target;
        target.spawn((Realm::Processor::TaskFuncID)
                         GlobalTaskId::RESHARD_POINT_TO_POINT_NAME_EXCHANGE,
                     &args, sizeof(args));
      }

      if (reshard.pending_dest_reshards.empty()) {
        const Realm::UserEvent pending = Realm::UserEvent::create_user_event();
        reshard.pending_source_reshards.push({PointToPointReshard::PendingOp{
            .precondition = src.Precondition(),
            .postcondition_to_trigger = pending,
            .partner_instance = src->tile().instance(),
        }});
        log_reshard.debug()
            << "on proc " << p.global_id()
            << " no pending reshards for source, return pending=" << pending;
        return (Realm::Event)pending;
      }

      PointToPointReshard::PendingOp& op =
          reshard.pending_dest_reshards.front();
      const Realm::Event postcondition = [&] {
        if (p.RealmProc().address_space() != target.address_space()) {
          return RemoteCopyNextPendingToTarget(
              reshard, src->tile().instance(), op.partner_instance,
              op.precondition, src.Precondition());
        }
        Realm::Event precondition =
            Realm::Event::merge_events(op.precondition, src.Precondition());
        Realm::Event postcondition =
            IssueCopy(reshard.shape, reshard.field_size, src->tile().instance(),
                      op.partner_instance, precondition);
        assert(!reshard.ack_done_barrier.exists());
        log_reshard.debug() << "on proc " << p.global_id()
                            << " found matching local reshard for source, "
                               "triggering postcondition="
                            << op.postcondition_to_trigger;
        op.postcondition_to_trigger.trigger(postcondition);
        return postcondition;
      }();
      reshard.pending_dest_reshards.pop();
      return postcondition;
    }();
    if (reshard.name.has_value()) {
      ProfileRegion(p, *reshard.name, src.Precondition(), postcondition);
    }
    mutex_.unlock();
    return postcondition;
  }
}

Realm::Event ReshardManager::ReshardToLocalTarget(
    const Processor& p, const RealmShape& shape,
    const ShardedArray::ReshardKey& key, const ShardedArray& src,
    const Store<ShardedArray>& dst) {
  switch (GetProtocol(dst->tile())) {
    case Protocol::EAGER:
      return EagerReshardToLocalTarget(p, shape, key, src, dst);
    case Protocol::RENDEZVOUS:
      return RendezvousReshardToLocalTarget(p, shape, key, src, dst);
  }
  // make the compiler happy
  return Realm::Event::NO_EVENT;
}

Realm::Event ReshardManager::EagerReshardToLocalTarget(
    const Processor& p, const RealmShape& shape,
    const ShardedArray::ReshardKey& key, const ShardedArray& src,
    const Store<ShardedArray>& dst) {
  throw std::runtime_error(
      "ReshardManager::EagerReshardToLocalTarget: not implemented");
}

Realm::Event ReshardManager::RendezvousReshardToLocalTarget(
    const Processor& p, const RealmShape& shape,
    const ShardedArray::ReshardKey& key, const ShardedArray& src,
    const Store<ShardedArray>& dst) {
  log_reshard.debug() << "RDVZ: recv " << dst->shape() << " shards "
                      << key.src_shard_id << "-> " << key.dst_shard_id << " on "
                      << p.global_id();

  // Check in read only mode to see if we've got all the barriers
  mutex_.wrlock().wait();
  PointToPointReshard& reshard = point_to_point_reshards_[key];
  if (!reshard.exists) {
    reshard.field_size = SupportedTypeSizeOf(src.shape().type);
    reshard.shape = shape;
    reshard.name = ReshardName(src, *dst);
    reshard.exists = true;
  }

  if (!reshard.pending_source_reshards.empty()) {
    PointToPointReshard::PendingOp& op =
        reshard.pending_source_reshards.front();
    Realm::Event precondition =
        Realm::Event::merge_events(op.precondition, dst.Precondition());
    Realm::Event postcondition =
        IssueCopy(reshard.shape, reshard.field_size, op.partner_instance,
                  dst->tile().instance(), precondition);
    assert(op.postcondition_to_trigger.exists());
    log_reshard.debug() << "on proc " << p.global_id()
                        << " target found matching source, copy issued, "
                           "triggering postcondition "
                        << op.postcondition_to_trigger;
    op.postcondition_to_trigger.trigger(postcondition);
    reshard.pending_source_reshards.pop();
    mutex_.unlock();
    return postcondition;
  }

  Realm::Processor target = src.HostForShard(key.src_shard_id);
  if (target.address_space() != p.RealmProc().address_space()) {
    if (!reshard.target_ready_barrier.exists()) {
      reshard.target_ready_barrier = Realm::Barrier::create_barrier(1);
    }

    PointToPointNameArgs args(key, reshard.target_ready_barrier,
                              dst->tile().instance());
    target.spawn((Realm::Processor::TaskFuncID)
                     GlobalTaskId::RESHARD_POINT_TO_POINT_NAME_EXCHANGE,
                 &args, sizeof(args));

    log_reshard.debug() << "target " << p.global_id() << " will trigger ready "
                        << reshard.target_ready_barrier << " after "
                        << dst.Precondition();

    reshard.target_ready_barrier.arrive(1, dst.Precondition());
    reshard.target_ready_barrier =
        reshard.target_ready_barrier.advance_barrier();
  }

  if (!reshard.ack_done_barrier.exists()) {
    // this is either a local exchange or we need the barrier to arrive
    // before we can wait on it
    Realm::UserEvent pending = Realm::UserEvent::create_user_event();
    log_reshard.debug() << "on proc " << p.global_id()
                        << " target has no matching source, waiting on trigger "
                        << pending;
    reshard.pending_dest_reshards.push(PointToPointReshard::PendingOp{
        .precondition = dst.Precondition(),
        .postcondition_to_trigger = pending,
        .partner_instance = dst->tile().instance(),
    });
    mutex_.unlock();
    return pending;
  }

  Realm::Event postcondition = reshard.ack_done_barrier;
  reshard.ack_done_barrier = reshard.ack_done_barrier.advance_barrier();
  mutex_.unlock();
  return postcondition;
}

}  // namespace zuku