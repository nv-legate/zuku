/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef _POC_SRC_REALM_VARIANTS_H_
#define _POC_SRC_REALM_VARIANTS_H_

#include "realm.h"

#include "type_traits.h"
#include <variant>

namespace zuku {

template <template <int, typename> typename RealmType, typename T>
using RealmDimVariant = std::variant<RealmType<1, T>, RealmType<2, T>,
                                     RealmType<3, T>, RealmType<4, T>
#if REALM_MAX_DIM >= 5
                                     ,
                                     RealmType<5, T>
#endif
                                     >;

using RealmTypeVariant = std::variant<int8_t, int16_t, int32_t, int64_t>;

using RealmShape = RealmDimVariant<Realm::Rect, long long>;

template <int N>
using RealmRect = Realm::Rect<N, long long>;

}  // namespace zuku

namespace std {

// zuku Realm shape is type alias for std::variant
inline std::ostream& operator<<(std::ostream& os,
                                const zuku::RealmShape& shape) {
  std::visit(zuku::overloaded{[&](const auto& rect) { os << rect; }}, shape);
  return os;
}

}  // namespace std

#endif