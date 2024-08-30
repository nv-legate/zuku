/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef _POC_SRC_TILED_ARRAY_H_
#define _POC_SRC_TILED_ARRAY_H_

#include "realm.h"
#include "realm/memory.h"
#include "realm/processor.h"

#include <optional>
#include <vector>

#include "hash.h"
#include "instance_cache.h"
#include "processor.h"
#include "realm_variants.h"
#include "shape.h"
#include "store.h"

namespace zuku {

struct TileShape {
  SupportedType type;
  std::vector<int64_t> dims;
  RealmShape bounds;
};
inline bool operator==(const TileShape& lhs, const TileShape& rhs) {
  return lhs.type == rhs.type && lhs.dims == rhs.dims;
}

inline bool operator!=(const TileShape& lhs, const TileShape& rhs) {
  return !(lhs == rhs);
}
template <typename H>
H AbslHashValue(H h, const TileShape& shape) {
  return H::combine(std::move(h), (int)shape.type, shape.dims);
}
inline std::ostream& operator<<(std::ostream& os, const TileShape& shape) {
  os << "TileShape(type=" << int(shape.type) << ",dims={";
  for (int64_t dim : shape.dims) {
    os << dim << ",";
  }
  os << "})";
  return os;
}

struct AllocationFailure : public std::runtime_error {
  explicit AllocationFailure(const ShardedShape& shape);

  explicit AllocationFailure(const TileShape& shape)
      : std::runtime_error("failed allocating tile") {}
};

std::vector<int64_t> ComputeTileIndex(int64_t index, const Sharding& sharding);

RealmShape ComputeTileBounds(int64_t shard, const Sharding& sharding);

int64_t ComputeTileShard(const std::vector<int64_t>& indices,
                         const Sharding& sharding);

int64_t ComputeRealmShapeSize(const RealmShape& shape);

class ArrayCache;

class ArrayTile {
 public:
  struct Ctor {
    std::optional<Realm::Memory::Kind> memory;
    std::optional<Processor> processor;
  };

  ArrayTile(ArrayTile&& a)
      : shape_(std::move(a.shape_)),
        instance_(a.instance_),
        type_(std::move(a.type_)) {
    // I hate that I have to do this, but
    // I need the instance to be marked
    // const to indicate that it can only
    // be initialized in the constructor
    const_cast<Realm::RegionInstance&>(a.instance_) =
        Realm::RegionInstance::NO_INST;
  }

  ArrayTile(const ArrayTile& a) = default;

  ~ArrayTile() = default;

  void DeferredDeleteInstances(const Realm::Event& ev);

  static Store<ArrayTile> CreateStore(TileShape shape, Ctor ctor) {
    auto [event, array] = Create(std::move(shape), std::move(ctor));
    return Store<ArrayTile>::DeferredCreate(std::move(event), std::move(array));
  }

  size_t byte_size() const;

  int64_t num_elements() const;

  const void* data() const;

  void* data();

  template <class T>
  const T* ptr() const {
    return static_cast<const T*>(data());
  }

  template <class T>
  T* ptr() {
    return static_cast<T*>(data());
  }

  static Future<ArrayTile> CreateFuture(TileShape shape, Ctor ctor);

  static std::pair<Realm::Event, ArrayTile> Create(TileShape shape, Ctor ctor);

  const Realm::RegionInstance& instance() const { return instance_; }

  const TileShape& shape() const { return shape_; }

  const RealmShape& realm_shape() const { return shape_.bounds; }

 private:
  ArrayTile(Realm::RegionInstance instance, TileShape shape,
            RealmTypeVariant type)
      : instance_(instance), shape_(std::move(shape)), type_(std::move(type)) {}

  static RealmTypeVariant ToRealmTypeVariant(SupportedType type);

  const void* InstancePointer() const;

 private:
  const Realm::RegionInstance instance_;
  const TileShape shape_;
  const RealmTypeVariant type_;
};

class ShardedArray {
 public:
  struct UniqueId {
    uint64_t id;
    DeviceList devices;

    bool operator==(const UniqueId& rhs) const {
      return id == rhs.id && devices == rhs.devices;
    }
  };

  struct ReshardKey {
    // tensor IDs are unique to a device
    // mesh the same ID
    const ShardedArray::UniqueId src_id;
    const ShardedArray::UniqueId dst_id;
    const int64_t src_shard_id;
    const int64_t dst_shard_id;

    bool operator==(const ReshardKey rhs) const {
      return src_id == rhs.src_id && dst_id == rhs.dst_id &&
             src_shard_id == rhs.src_shard_id &&
             dst_shard_id == rhs.dst_shard_id;
    }
  };

  struct ScatterGatherKey {
    const ShardedArray::UniqueId src_id;
    const ShardedArray::UniqueId dst_id;

    inline bool operator==(const ScatterGatherKey& rhs) const {
      return (src_id == rhs.src_id) && (dst_id == rhs.dst_id);
    }
  };

  struct Ctor {
    std::optional<uint64_t> assigned_id{std::nullopt};
    std::optional<std::string> name{std::nullopt};
    std::optional<int64_t> shard_id{std::nullopt};
    Processor processor;
    std::optional<Realm::Memory::Kind> memory{std::nullopt};
  };

  ShardedArray(const ShardedArray&) = delete;
  ShardedArray(ShardedArray&&) = default;
  ~ShardedArray();

  const ShardedShape& shape() const { return shape_; }

  int64_t shard_id() const { return shard_id_; }

  bool HasName() const { return name_.has_value(); }

  const std::string& name() const { return *name_; }

  UniqueId unique_id() const {
    return {.id = mesh_unique_id_, .devices = shape_.sharding.devices};
  }

  uint64_t mesh_unique_id() const { return mesh_unique_id_; }

  ArrayTile& tile() & { return *array_; }

  const ArrayTile& tile() const& { return *array_; }

  bool HasTile() const { return array_.has_value(); }

  void DeferredDeleteInstances(const Realm::Event& ev);

  const Processor& processor() const { return proc_; }

  static void MoveToMemory(
      Store<ShardedArray>& src, Store<ShardedArray>& dst,
      std::optional<Realm::Event> precondition = std::nullopt);

  static Store<ShardedArray> MoveToMemory(
      Store<ShardedArray>& src, ArrayCache& cache,
      std::optional<Realm::Event> precondition = std::nullopt);

  static Store<ShardedArray> MoveToMemory(
      Store<ShardedArray> src, ArrayCache& save_array_cache,
      ArrayCache& new_array_cache,
      std::optional<Realm::Event> precondition = std::nullopt);

  Realm::Processor HostForShard(int64_t shard_id) const;

  static Store<ShardedArray> Create(ShardedShape shape, Ctor ctor);

  ShardedArray(ShardedShape shape, Ctor ctor, std::optional<ArrayTile> array)
      : array_(std::move(array)),
        shape_(std::move(shape)),
        shard_id_(ctor.shard_id.value_or(-1)),
        proc_(ctor.processor),
        mesh_unique_id_(
            AllocateNextGlobalId(ctor.processor, shape, ctor.assigned_id)),
        name_(std::move(ctor.name)) {}

 private:
  void ReassignId(uint64_t id);

  static uint64_t AllocateNextGlobalId(
      const Processor& p, const ShardedShape& shape,
      std::optional<uint64_t> assigned_id = std::nullopt);

  const volatile uint64_t mesh_unique_id_;
  const ShardedShape shape_;
  const int64_t shard_id_;
  const Processor proc_;
  const std::optional<std::string> name_;
  std::optional<ArrayTile> array_;
};

std::ostream& operator<<(std::ostream& os, const ShardedArray::UniqueId& id);

std::ostream& operator<<(std::ostream& os,
                         const zuku::ShardedArray::ReshardKey& key);

class ArrayCache : public InstanceCache<ShardedShape, ShardedArray> {
 public:
  ArrayCache(Realm::Memory::Kind memory, Processor p)
      : InstanceCache(
            std::move(memory), std::move(p),
            [](ShardedShape shape, Realm::Memory::Kind memory, Processor p) {
              return ShardedArray::Create(shape,
                                          {.processor = p, .memory = memory});
            }) {}
};

}  // namespace zuku

std::ostream& operator<<(std::ostream& os, const zuku::RealmShape& shape);

template <>
struct std::hash<zuku::ShardedArray::UniqueId> {
  std::size_t operator()(const zuku::ShardedArray::UniqueId& key) const {
    return zuku::hash(key.devices, key.id);
  }
};

template <>
struct std::hash<zuku::ShardedArray::ReshardKey> {
  std::size_t operator()(const zuku::ShardedArray::ReshardKey& key) const {
    return zuku::hash(key.src_id, key.dst_id, key.src_shard_id,
                      key.dst_shard_id);
  }
};

template <>
struct std::hash<zuku::ShardedArray::ScatterGatherKey> {
  std::size_t operator()(
      const zuku::ShardedArray::ScatterGatherKey& key) const {
    return zuku::hash(key.src_id, key.dst_id);
  }
};

template <>
struct std::hash<zuku::TileShape> {
  std::size_t operator()(const zuku::TileShape& shape) const {
    return zuku::hash(shape.dims, shape.type);
  }
};

#endif
