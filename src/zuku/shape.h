/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef _POC_SRC_SHAPE_H_
#define _POC_SRC_SHAPE_H_

#include <stdexcept>
#include <vector>

#include "mesh.h"

namespace zuku {

enum class SupportedType {
  PRED,
  F16,
  BF16,
  F32,
  F64,
  S8,
  S16,
  S32,
  S64,
  U8,
  U16,
  U32,
  U64,
  C64,
  C128,
};

inline std::size_t SupportedTypeSizeOf(SupportedType type) {
  size_t bytesize = 0;
  switch (type) {
    case SupportedType::PRED:
    case SupportedType::S8:
    case SupportedType::U8:
      bytesize = 1;
      break;
    case SupportedType::F16:
    case SupportedType::BF16:
    case SupportedType::S16:
    case SupportedType::U16:
      bytesize = 2;
      break;
    case SupportedType::F32:
    case SupportedType::S32:
    case SupportedType::U32:
      bytesize = 4;
      break;
    case SupportedType::F64:
    case SupportedType::S64:
    case SupportedType::U64:
    case SupportedType::C64:
      bytesize = 8;
      break;
    case SupportedType::C128:
      bytesize = 16;
      break;
    default:
      throw std::runtime_error("unsupported SupportedType");
      break;
  }
  return bytesize;
}

#define supported_type_case(x) \
  case SupportedType::x:       \
    return #x
inline const char* ToString(SupportedType type) {
  switch (type) {
    supported_type_case(PRED);
    supported_type_case(F16);
    supported_type_case(BF16);
    supported_type_case(F32);
    supported_type_case(F64);
    supported_type_case(S8);
    supported_type_case(S16);
    supported_type_case(S32);
    supported_type_case(S64);
    supported_type_case(U8);
    supported_type_case(U16);
    supported_type_case(U32);
    supported_type_case(U64);
    supported_type_case(C64);
    supported_type_case(C128);
  }
  return "NONE";
}
#undef supported_type_case

struct ShardingDim {
  int64_t size;
  int64_t sharding;
  int64_t permutation{0};
};
inline bool operator==(const ShardingDim& lhs, const ShardingDim& rhs) {
  return lhs.size == rhs.size && lhs.sharding == rhs.sharding &&
         lhs.permutation == rhs.permutation;
}
inline bool operator!=(const ShardingDim& lhs, const ShardingDim& rhs) {
  return !(lhs == rhs);
}
template <typename H>
H AbslHashValue(H h, const ShardingDim& dim) {
  return H::combine(std::move(h), dim.size, dim.sharding, dim.permutation);
}
inline std::ostream& operator<<(std::ostream& os, const ShardingDim& dim) {
  os << "ShardingDim(size=" << dim.size << ",shard=" << dim.sharding
     << ",perm=" << dim.permutation << ")";
  return os;
}

struct Sharding {
  std::vector<ShardingDim> dims;
  DeviceList devices;

  int64_t TotalSharding() const {
    int64_t sharding = 1;
    for (auto&& dim : dims) {
      // a replicate dim has size 1 and non-unit sharding
      if (dim.size > 1) {
        sharding *= dim.sharding;
      }
    }
    return sharding;
  }

  // the number of actual dimensions, not counting
  // replication dims that are not part of the real shape
  int64_t NumRealDims() const {
    for (const auto& dim : dims) {
      if (dim.size < dim.sharding) {
        return dims.size() - 1;
      }
    }
    return dims.size();
  }

  int64_t NumLocalElements() const {
    int64_t size = 1;
    for (auto&& dim : dims) {
      // a replicate dim has size 1 and non-unit sharding
      if (dim.size > 1) {
        size *= dim.size / dim.sharding;
      }
    }
    return size;
  }

  bool IsPartiallyReplicated() const {
    const int64_t sharding = TotalSharding();
    return sharding > 1 && sharding < devices.size();
  }

  bool FullySharded() const { return TotalSharding() == devices.size(); }

  bool IsReplicated() const { return TotalSharding() == 1; }
};
inline bool operator==(const Sharding& lhs, const Sharding& rhs) {
  return lhs.dims == rhs.dims && lhs.devices == rhs.devices;
}
inline bool operator!=(const Sharding& lhs, const Sharding& rhs) {
  return !(lhs == rhs);
}
inline std::ostream& operator<<(std::ostream& os, const Sharding& sharding) {
  os << "Sharding(" << sharding.devices << ",dims={";
  for (auto&& dim : sharding.dims) {
    os << dim << ",";
  }
  os << "})";
  return os;
}
template <typename H>
H AbslHashValue(H h, const Sharding& sharding) {
  return H::combine(std::move(h), sharding.dims, sharding.devices);
}
struct ShardedShape {
  SupportedType type;
  Sharding sharding;
};

inline bool operator==(const ShardedShape& lhs, const ShardedShape& rhs) {
  return lhs.type == rhs.type && lhs.sharding == rhs.sharding;
}

inline bool operator!=(const ShardedShape& lhs, const ShardedShape& rhs) {
  return !(lhs == rhs);
}

template <typename H>
H AbslHashValue(H h, const ShardedShape& shape) {
  return H::combine(std::move(h), (int)shape.type, shape.sharding);
}

inline std::ostream& operator<<(std::ostream& os, const ShardedShape& shape) {
  os << "ShardedShape(type=" << int(shape.type) << "," << shape.sharding << ")";
  return os;
}

inline std::ostream& ShortString(std::ostream& os, const ShardedShape& shape) {
  os << "TYPE=" << ToString(shape.type);
  os << " [ ";
  for (const auto& dim : shape.sharding.dims) {
    os << dim.size << " ";
  }
  os << "] % [ ";
  for (const auto& dim : shape.sharding.dims) {
    os << dim.sharding << " ";
  }
  os << "]";
  return os;
}

inline int64_t ShardElements(const Sharding& sharding) {
  int64_t dim_product = 1;
  for (auto& dim : sharding.dims) {
    // don't divide by replicated sharding
    if (dim.size > 1) {
      dim_product *= dim.size;
      dim_product /= dim.sharding;
    }
  }
  return dim_product;
}

inline int64_t ShardSize(const ShardedShape& shape) {
  return SupportedTypeSizeOf(shape.type) * ShardElements(shape.sharding);
}

inline int64_t FullDimensionProduct(const ShardedShape& shape) {
  int64_t dim_product = 1;
  for (auto& dim : shape.sharding.dims) {
    dim_product *= dim.size;
  }
  return dim_product;
}

}  // namespace zuku

template <>
struct std::hash<zuku::ShardingDim> {
  std::size_t operator()(const zuku::ShardingDim& dim) const {
    return zuku::hash(dim.sharding, dim.size, dim.permutation);
  }
};

template <>
struct std::hash<zuku::Sharding> {
  std::size_t operator()(const zuku::Sharding& sharding) const {
    return zuku::hash(sharding.dims, sharding.devices);
  }
};

template <>
struct std::hash<zuku::ShardedShape> {
  std::size_t operator()(const zuku::ShardedShape& shape) const {
    return zuku::hash(shape.sharding, shape.type);
  }
};

#endif
