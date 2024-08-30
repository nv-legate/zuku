/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "realm_variants.h"
#include "type_traits.h"
#include "tiled_array.h"

namespace zuku {

inline int UniqueValue(int i) { return i; }

inline int UniqueValue(int i, int j) { return i * 100 + j; }

inline int UniqueValue(int i, int j, int k) { return i * 10000 + j * 100 + k; }

inline int UniqueValue(int i, int j, int k, int l) {
  return i * 10000 + j * 100 + k;
}

inline int UniqueValue(int i, int j, int k, int l, int m) {
  return i * 10000 + j * 100 + k;
}

template <int DIM, typename DATA, int N, typename T, typename Lambda,
          typename... Args>
void _Iterate(DATA* data, const Realm::Rect<N, T>& iter_rect,
              const Realm::Rect<N, T>& total_rect, Lambda& lambda, int stride,
              int index, Args&&... args) {
  if constexpr (DIM == -1) {
    lambda(data[index], std::forward<Args>(args)...);
  } else {
    for (int i = iter_rect.lo[DIM]; i <= iter_rect.hi[DIM]; ++i) {
      const int dim_size = total_rect.hi[DIM] - total_rect.lo[DIM] + 1;
      _Iterate<DIM - 1>(data, iter_rect, total_rect, lambda, stride * dim_size,
                        (i - total_rect.lo[DIM]) * stride + index,
                        std::forward<Args>(args)..., i);
    }
  }
}

template <typename DATA, int N, typename T, typename Lambda>
void Iterate(DATA* data, const Realm::Rect<N, T>& iter_rect,
             const Realm::Rect<N, T>& total_rect, Lambda& lambda) {
  _Iterate<N - 1>(data, iter_rect, total_rect, lambda, 1, 0);
}

template <typename DATA, typename Lambda>
void Iterate(DATA* data, const RealmShape& iter_bounds,
             const RealmShape& total_bounds, Lambda&& lambda) {
  std::visit(overloaded{[&](const auto& iter_rect) {
               std::visit(overloaded{[&](const auto& total_rect) {
                            Iterate(data, iter_rect, total_rect, lambda);
                          }},
                          total_bounds);
             }},
             iter_bounds);
}

template <typename Lambda>
void Iterate(ArrayTile& tile, Lambda&& lambda) {
  int* data = tile.ptr<int>();
  Iterate(data, tile.shape().bounds, tile.shape().bounds,
          std::forward<Lambda>(lambda));
}

template <typename Lambda>
void Iterate(const ArrayTile& tile, Lambda&& lambda) {
  const int* data = tile.ptr<int>();
  Iterate(data, tile.shape().bounds, tile.shape().bounds,
          std::forward<Lambda>(lambda));
}

template <int N, int M, typename T>
bool RectContains(const Realm::Rect<N, T>& outer,
                  const Realm::Rect<M, T>& inner) {
  if constexpr (N != M) {
    return false;
  } else {
    for (int dim = 0; dim < N; ++dim) {
      if (inner.lo[dim] < outer.lo[dim] || inner.hi[dim] > outer.hi[dim]) {
        return false;
      }
    }
    return true;
  }
}

inline bool ShapeContains(const RealmShape& outer, const RealmShape& inner) {
  return std::visit(overloaded{[&](const auto& outer_rect) {
                      return std::visit(overloaded{[&](const auto& inner_rect) {
                                          return RectContains(outer_rect,
                                                              inner_rect);
                                        }},
                                        inner);
                    }},
                    outer);
}

}  // namespace zuku
