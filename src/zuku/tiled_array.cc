/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "tiled_array.h"

#include "realm.h"
#include "realm/event.h"
#include "realm/inst_layout.h"
#include "realm/logging.h"
#include "realm/processor.h"

#include <mutex>
#include <stdexcept>

#include "hash.h"
#include "init.h"
#include "processor.h"
#include "profile.h"
#include "realm_variants.h"
#include "shape.h"

Realm::Logger log_tile("zuku-tile");

namespace zuku {
namespace {

struct GlobalIdKey {
  DeviceList devices;
  const int64_t local_proc_id;
  Processor::Type proc_type;

  bool operator==(const GlobalIdKey& key) const {
    return key.devices == devices && local_proc_id == key.local_proc_id &&
           key.proc_type == proc_type;
  }
};

template <int N>
RealmShape ComputeTileBounds(int64_t shard, const Sharding& sharding) {
  RealmRect<N> rect;
  if (sharding.dims.empty() ||
      (sharding.dims.size() == 1 && sharding.dims.front().size == 1)) {
    rect.lo[0] = 0;
    rect.hi[0] = 0;
    return rect;
  }

  int64_t dim_stride = 1;
  for (int64_t dim = 0; dim < sharding.dims.size(); ++dim) {
    dim_stride *= sharding.dims[dim].sharding;
  }

  std::vector<int64_t> index = ComputeTileIndex(shard, sharding);

  const int64_t replication = sharding.devices.size() / dim_stride;
  int64_t remainder = shard / replication;
  int64_t rect_dim = 0;
  for (int64_t dim = 0; dim < sharding.dims.size(); ++dim) {
    // fake sharding dimension, do not include
    if (sharding.dims[dim].size == 1 && sharding.dims[dim].sharding > 1) {
      continue;
    }
    const int64_t tile_extent =
        sharding.dims[dim].size / sharding.dims[dim].sharding;
    const int64_t start = tile_extent * index[dim];
    const int64_t stop = start + tile_extent - 1;
    rect.lo[rect_dim] = start;
    rect.hi[rect_dim] = stop;
    ++rect_dim;
  }
  return rect;
}

template <int N, typename T>
int64_t ComputeRealmShapeSize(const Realm::Rect<N, T>& rect) {
  int64_t size = 1;
  for (int i = 0; i < N; ++i) {
    size *= (rect.hi[i] - rect.lo[i] + 1);
  }
  return size;
}

}  // namespace
}  // namespace zuku

template <>
struct std::hash<zuku::GlobalIdKey> {
  std::size_t operator()(const zuku::GlobalIdKey& key) const {
    return zuku::hash(key.devices, key.local_proc_id, (int)key.proc_type);
  }
};

namespace zuku {
namespace {

std::string AllocationFailureMessage(const ShardedShape& shape) {
  std::stringstream sstr;
  sstr << "Failed to allocate sharded shape " << shape;
  return sstr.str();
}

#ifdef ZUKU_USE_TILE_CACHE
auto& TileCache() {
  static std::unordered_map<
      Processor, std::unordered_map<
                     TileShape, std::deque<std::pair<Realm::Event, ArrayTile>>>>
      tiles;
  return tiles;
}
#endif

std::optional<std::pair<Realm::Event, ArrayTile>> GetTile(
    const TileShape& shape, const Processor& p) {
#ifdef ZUKU_USE_TILE_CACHE
  auto& cache = TileCache()[p];
  auto iter = cache.find(shape);
  if (iter == cache.end() || iter->second.empty()) {
    return std::nullopt;
  }
  auto pair = std::move(iter->second.front());
  iter->second.pop_front();
  return pair;
#else
  return std::nullopt;
#endif
}

void FreeTile(ArrayTile tile, Realm::Event free, const Processor& p) {
#ifdef ZUKU_USE_TILE_CACHE
  auto& cache = TileCache()[p];
  cache[tile.shape()].emplace_back(free, std::move(tile));
#else
  tile.DeferredDeleteInstances(free);
#endif
}

void ClearTiles(const Processor& p) {
#ifdef ZUKU_USE_TILE_CACHE
  auto& cache = TileCache()[p];
  for (auto& [shape, deq] : cache) {
    for (auto& [ev, tile] : deq) {
      tile.DeferredDeleteInstances(ev);
    }
    deq.clear();
  }
  cache.clear();
#endif
}

auto& AllocationCounts() {
  static std::unordered_map<Processor,
                            std::unordered_map<ShardedShape, int64_t>>
      allocation_counts;
  return allocation_counts;
}

}  // namespace

AllocationFailure::AllocationFailure(const ShardedShape& shape)
    : std::runtime_error(AllocationFailureMessage(shape)) {}

int64_t ComputeRealmShapeSize(const RealmShape& shape) {
  return std::visit(
      overloaded{[&](const auto& rect) { return ComputeRealmShapeSize(rect); }},
      shape);
}

int64_t ComputeTileShard(const std::vector<int64_t>& indices,
                         const Sharding& sharding) {
  if (indices.empty()) {
    return 0;
  }

  int64_t index = indices[0];
  for (int dim = 1; dim < indices.size(); ++dim) {
    index = (index * sharding.dims[dim].sharding) + indices[dim];
  }
  return index;
}

std::vector<int64_t> ComputeTileIndex(int64_t index, const Sharding& sharding) {
  // unpermute the sizes and compute the index for the unpermuted dims
  std::vector<int64_t> shard_extents(sharding.dims.size());
  std::vector<int64_t> permutation(sharding.dims.size());
  int p_dim = 0;
  for (const auto& dim : sharding.dims) {
    shard_extents[dim.permutation] = dim.sharding;
    permutation[dim.permutation] = p_dim++;
  }

  int64_t dim_stride = 1;
  for (int64_t extent : shard_extents) {
    dim_stride *= extent;
  }

  std::vector<int64_t> indices(sharding.dims.size());
  int64_t remainder = index;
  for (int64_t dim = 0; dim < shard_extents.size(); ++dim) {
    dim_stride /= shard_extents[dim];
    const int64_t dim_index = remainder / dim_stride;
    indices[permutation[dim]] = dim_index;
    remainder -= dim_index * dim_stride;
  }
  return indices;
}

RealmShape ComputeTileBounds(int64_t shard, const Sharding& sharding) {
  switch (sharding.NumRealDims()) {
    case 0:
      return ComputeTileBounds<1>(shard, sharding);
    case 1:
      return ComputeTileBounds<1>(shard, sharding);
    case 2:
      return ComputeTileBounds<2>(shard, sharding);
    case 3:
      return ComputeTileBounds<3>(shard, sharding);
    case 4:
      return ComputeTileBounds<4>(shard, sharding);
#if REALM_MAX_DIM >= 5
    case 5:
      return ComputeTileBounds<5>(shard, sharding);
#endif
    default:
      throw std::runtime_error("do not support arrays with " +
                               std::to_string(sharding.NumRealDims()) +
                               " dimensions");
      return Realm::Rect<1, long long>{};
  }
}

uint64_t ShardedArray::AllocateNextGlobalId(
    const Processor& p, const ShardedShape& shape,
    std::optional<uint64_t> assigned_id) {
  if (assigned_id.has_value()) {
    return *assigned_id;
  }

  static std::mutex lock;
  const std::lock_guard<std::mutex> lock_context(lock);
  static std::unordered_map<GlobalIdKey, uint64_t> next_ids;
  // the key depends on the processor since the same sharded array may be
  // created on multiple processors
  GlobalIdKey key{
      .devices = shape.sharding.devices,
      .local_proc_id = p.local_id(),
      .proc_type = p.type(),
  };
  return next_ids[key]++;
}

const void* ArrayTile::InstancePointer() const {
  return std::visit(
      overloaded{[&](auto& rect) {
        constexpr int N = rect_info<std::decay_t<decltype(rect)>>::N;
        using T = typename rect_info<std::decay_t<decltype(rect)>>::index_type;
        if (!instance_.exists()) {
          throw std::runtime_error(
              "fetching pointer from instance that doesn't exist");
        }
        return std::visit(
            overloaded{[&](auto type) -> void* {
              using FT = decltype(type);
              Realm::AffineAccessor<FT, N, T> acc{instance_, /*field_id=*/0};
              return acc.ptr(rect.lo);
            }},
            type_);
      }},
      shape_.bounds);
}

const void* ArrayTile::data() const { return InstancePointer(); }

void* ArrayTile::data() {
  // const-cast is always hideous, but better to avoid
  // code duplication on instance pointer fetching
  return const_cast<void*>(InstancePointer());
}

RealmTypeVariant ArrayTile::ToRealmTypeVariant(SupportedType type) {
  switch (type) {
    case SupportedType::PRED:
    case SupportedType::S8:
    case SupportedType::U8:
      return int8_t{};
    case SupportedType::F16:
    case SupportedType::BF16:
    case SupportedType::S16:
    case SupportedType::U16:
      return int16_t{};
    case SupportedType::F32:
    case SupportedType::S32:
    case SupportedType::U32:
      return int32_t{};
    case SupportedType::F64:
    case SupportedType::S64:
    case SupportedType::U64:
    case SupportedType::C64:
      return int64_t{};
    default:
      throw std::runtime_error("unsupported SupportedType");
      break;
  }
  // to suppress warnings
  return int8_t{};
}

size_t ArrayTile::byte_size() const {
  return num_elements() * SupportedTypeSizeOf(shape_.type);
}

int64_t ArrayTile::num_elements() const {
  size_t size = 1;
  for (auto&& dim : shape_.dims) {
    size *= dim;
  }
  return size;
}

Future<ArrayTile> ArrayTile::CreateFuture(TileShape shape, Ctor ctor) {
  auto [event, tile] = Create(std::move(shape), std::move(ctor));
  return Future<ArrayTile>{std::move(event), std::move(tile)};
}

void ArrayTile::DeferredDeleteInstances(const Realm::Event& ev) {
  if (instance_.exists()) {
    log_tile.debug() << "deleting instance " << instance_ << " after " << ev;
    instance_.destroy(ev);
  }
}

template <int N, typename T>
std::tuple<Realm::Event, Realm::RegionInstance, bool> TryAllocation(
    const Realm::Rect<N, T>& rect, Realm::Memory memory,
    const Realm::InstanceLayoutConstraints& constraints) {
  Realm::RegionInstance instance;
  if constexpr (N > 0) {
    // Use C order for dimensions for now
    int order[N];
    for (unsigned idx = 0; idx < N; idx++) order[idx] = (N - (idx + 1));
    const Realm::IndexSpace<N, T> bounds(rect);
    Realm::InstanceLayoutGeneric* layout =
        Realm::InstanceLayoutGeneric::choose_instance_layout<N, T>(
            bounds, constraints, order);

    Realm::ProfilingRequestSet requests;
    AllocationResult alloc{};
    AllocationResult* user_arg = &alloc;
    Realm::ProfilingRequest& before =
        requests.add_request(Processor::Util().RealmProc(),
                             (int)GlobalTaskId::CHECK_MEMORY_ALLOCATION,
                             &user_arg, sizeof(AllocationResult*));
    before.add_measurement<Realm::ProfilingMeasurements::InstanceAllocResult>();
    Realm::Event ready = Realm::RegionInstance::create_instance(
        instance, memory, layout, requests);
    alloc.Wait();
    log_tile.debug() << "created tile " << instance << " with shape " << bounds
                     << " on memory kind " << memory.kind();
    return std::make_tuple(std::move(ready), std::move(instance),
                           alloc.Succeeded());
  } else {
    Realm::Event ready = Realm::Event::NO_EVENT;
    return std::make_tuple(std::move(ready), std::move(instance), true);
  }
}

std::pair<Realm::Event, ArrayTile> ArrayTile::Create(TileShape shape,
                                                     Ctor ctor) {
  Processor p = [&] {
    if (ctor.processor.has_value()) {
      return *std::move(ctor.processor);
    }
    return Processor::Default();
  }();

  auto cached_tile = GetTile(shape, p);
  if (cached_tile.has_value()) {
    log_tile.debug() << "found cached tile " << cached_tile->second.instance_
                     << " for shape " << shape;
    return *std::move(cached_tile);
  }

  RealmTypeVariant realm_type = ToRealmTypeVariant(shape.type);
  const std::vector<Realm::FieldID> field_ids(1, 0);  // field ID zero
  const size_t field_size = SupportedTypeSizeOf(shape.type);
  const std::vector<size_t> field_sizes(1, field_size);
  // This is not the preferred way of making instance layouts anymore, but it's
  // the most expedient way of doing it so that is what we're going to do for
  // now
  const Realm::InstanceLayoutConstraints constraints(field_ids, field_sizes,
                                                     0 /*SOA*/);

  Realm::Memory memory = [&] {
    Realm::Machine::MemoryQuery mq(Realm::Machine::get_machine());
    mq.has_affinity_to(p.RealmProc());
    if (ctor.memory.has_value()) {
      mq.only_kind(*ctor.memory);
    }
    return mq.first();
  }();

  auto [event, instance] =
      std::visit(overloaded{[&](auto& rect) {
                   {
                     auto [event, instance, success] =
                         TryAllocation(rect, memory, constraints);
                     if (success) {
                       log_tile.debug() << "created new instance " << instance
                                        << " for rectange " << rect
                                        << " after event=" << event;
                       return std::make_pair(event, instance);
                     }
                   }

                   // We weren't able to reuse a tile from the cache and
                   // couldn't allocate from new memory. Clear the tile cache to
                   // free up some memory
                   ClearTiles(p);

                   auto [event, instance, success] =
                       TryAllocation(rect, memory, constraints);
                   if (success) {
                     return std::make_pair(event, instance);
                   }

                   // no tiles to reuse, no new memory space, and clearing the
                   // cache didn't make enough room
                   throw AllocationFailure(shape);

                   // make the compiler happy
                   return std::make_pair(Realm::Event::NO_EVENT,
                                         Realm::RegionInstance::NO_INST);
                 }},
                 shape.bounds);

  const auto type_variant = ToRealmTypeVariant(shape.type);
  return {std::move(event), ArrayTile{std::move(instance), std::move(shape),
                                      std::move(type_variant)}};
}

Realm::Processor ShardedArray::HostForShard(int64_t shard_id) const {
  const int64_t global_device_id = shard_id + shape_.sharding.devices.start();
  return Processor::GetAttachedHost(global_device_id, proc_.type());
}

Store<ShardedArray> ShardedArray::Create(ShardedShape shape, Ctor ctor) {
  int64_t total_shards = 1;
  for (const auto& dim : shape.sharding.dims) {
    total_shards *= dim.sharding;
  }
  if (total_shards != shape.sharding.devices.size()) {
    std::stringstream sstr;
    sstr << "sharding has total sharding = " << total_shards
         << " that differs from number of devices in mesh: " << shape.sharding;
    throw std::runtime_error(sstr.str());
  }

  if (!ctor.shard_id.has_value()) {
    ctor.shard_id = [&]() -> std::optional<int64_t> {
      if (shape.sharding.devices.Contains(ctor.processor.global_id())) {
        return ctor.processor.global_id() - shape.sharding.devices.start();
      }
      return std::nullopt;
    }();
    if (!ctor.shard_id.has_value()) {
      // if still no shard id, then there is no local shard
      return Store<ShardedArray>::Create(std::move(shape), std::move(ctor),
                                         std::nullopt);
    }
  }

  std::vector<int64_t> tiled_dims;
  tiled_dims.reserve(shape.sharding.dims.size());
  for (auto&& dim : shape.sharding.dims) {
    if (dim.size >= dim.sharding) {
      tiled_dims.push_back(dim.size / dim.sharding);
    }
  }
  TileShape tiled_shape{
      .type = shape.type,
      .dims = std::move(tiled_dims),
      .bounds = ComputeTileBounds(*ctor.shard_id, shape.sharding)};

  if (log_tile.want_debug()) {
    std::visit(
        [&](const auto& rect) {
          log_tile.debug() << "computed tile bounds " << rect << " for shard "
                           << *ctor.shard_id << " for sharding "
                           << shape.sharding;
        },
        tiled_shape.bounds);
  }

  try {
    auto [event, array] = ArrayTile::Create(
        std::move(tiled_shape),
        {.memory = std::move(ctor.memory), .processor = ctor.processor});
    AllocationCounts()[ctor.processor][shape]++;
    return Store<ShardedArray>::DeferredCreate(
        std::move(event), std::move(shape), std::move(ctor), std::move(array));
  } catch (AllocationFailure& failure) {
    std::stringstream sstr;
    int64_t total = 0;
    sstr << "Allocated arrays: type [shape] % [sharding]\n";
    for (const auto& [shape, count] : AllocationCounts()[ctor.processor]) {
      if (count > 0) {
        sstr << "Allocations: " << count << " x ";
        ShortString(sstr, shape) << "\n";
        total += ShardSize(shape) * count;
      }
    }
    sstr << "Total of allocations: " << (total / 1e9) << "GB\n";
    sstr << "Failed allocating new array " << shape << "\n";
    std::cerr << sstr.str() << std::endl;
    throw AllocationFailure(shape);
  }
  // keep compiler happy
  return Store<ShardedArray>::CreateEmpty();
}

ShardedArray::~ShardedArray() { AllocationCounts()[proc_][shape_]--; }

void ShardedArray::DeferredDeleteInstances(const Realm::Event& ev) {
  if (HasTile()) {
    FreeTile(*array_, ev, proc_);
  }
}

void ShardedArray::ReassignId(uint64_t id) {
  // TODO: temporary awfulness for uniquely identifying the sharded array
  // and caching unused arrays
  const_cast<volatile uint64_t&>(mesh_unique_id_) = id;
}

void ShardedArray::MoveToMemory(Store<ShardedArray>& src,
                                Store<ShardedArray>& dst,
                                std::optional<Realm::Event> precondition) {
  if (src->HasTile()) {
    const size_t field_size = SupportedTypeSizeOf(src->shape().type);
    std::vector<Realm::CopySrcDstField> srcs(1);
    srcs[0].set_field(src->tile().instance(), 0 /*field id*/, field_size);

    std::vector<Realm::CopySrcDstField> dsts(1);
    dsts[0].set_field(dst->tile().instance(), 0 /*field id*/, field_size);

    const Realm::Event combined_precondition = Realm::Event::merge_events(
        src.Precondition(), dst.Precondition(),
        precondition.value_or(Realm::Event::NO_EVENT));

    // enqueue a copy between the tiles
    const Realm::Event postcondition = std::visit(
        overloaded{[&](auto& bounds) {
          const Realm::ProfilingRequestSet no_requests;
          const int priority = 0;
          log_tile.debug() << "copying " << src->tile().instance() << " to "
                           << dst->tile().instance() << " on bounds " << bounds
                           << " from " << src->processor() << " to "
                           << dst->processor();
          return bounds.copy(srcs, dsts, no_requests, combined_precondition,
                             priority);
        }},
        src->tile().realm_shape());

    ProfileRegion(src->processor(), "copy", combined_precondition,
                  postcondition);

    src.ReadyAfter(postcondition);
    dst.ReadyAfter(postcondition);
  }

  dst->ReassignId(src->mesh_unique_id());
}

Store<ShardedArray> ShardedArray::MoveToMemory(
    Store<ShardedArray>& src, ArrayCache& cache,
    std::optional<Realm::Event> precondition) {
  Store<ShardedArray> new_array = cache.Get(src->shape());
  MoveToMemory(src, new_array, std::move(precondition));
  return new_array;
}

Store<ShardedArray> ShardedArray::MoveToMemory(
    Store<ShardedArray> src, ArrayCache& save_array_cache,
    ArrayCache& new_array_cache, std::optional<Realm::Event> precondition) {
  Store<ShardedArray> new_array =
      MoveToMemory(src, new_array_cache, std::move(precondition));
  ShardedShape shape = src->shape();
  save_array_cache.Free(shape, std::move(src));
  return new_array;
}

std::ostream& operator<<(std::ostream& os, const ShardedArray::UniqueId& id) {
  os << "UniqueId(" << id.devices << "," << id.id << ")";
  return os;
}

std::ostream& operator<<(std::ostream& os,
                         const ShardedArray::ReshardKey& key) {
  os << "ReshardKey(" << key.src_id << "," << key.dst_id << ","
     << key.src_shard_id << "," << key.dst_shard_id << ")";
  return os;
}

}  // namespace zuku

std::ostream& operator<<(std::ostream& os, const zuku::RealmShape& shape) {
  std::visit([&](const auto& rect) { os << rect; }, shape);
  return os;
}
