/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef _POC_SRC_RESOLVE_DEPENDENCY_H_
#define _POC_SRC_RESOLVE_DEPENDENCY_H_

#include <iostream>
#include <tuple>

#include "future.h"
#include "store.h"
#include "type_traits.h"
#include "vector.h"

namespace zuku {

template <class Src, class Target>
struct ResolveDependency {
  Src operator()(Src&& src) { return std::move(src); }
};

template <class T>
struct ResolveDependency<Future<T>, Future<T>> {
  Future<T> operator()(Future<T>&& src) { return std::move(src); }
};

template <class Src, class Target>
struct ResolveDependency<Future<Src>, Target> {
  Src operator()(Future<Src>&& src) { return std::move(src).get_value(); }
};

template <class T>
struct ResolveDependency<Store<T>, T&> {
  T& operator()(Store<T>& src) { return src.get_value(); }
};

template <class T>
struct ResolveDependency<Store<T>, const T&> {
  T& operator()(Store<T>& src) { return src.get_value(); }
};

template <class T>
struct ResolveDependency<Store<T>, Store<T>> {
  Store<T> operator()(Store<T>&& src) { return std::move(src); }
};

template <class T>
struct ResolveDependency<View<T>, View<T>> {
  View<T> operator()(View<T>&& src) { return std::move(src); }
};

template <class T>
struct ResolveDependency<View<T>, const T&> {
  const T& operator()(View<T>&& src) { return src.value(); }
};

template <class T>
struct ResolveDependency<View<T>, T> {
  const T& operator()(View<T>&& src) { return src.value(); }
};

template <class T>
struct ResolveDependency<std::vector<Store<T>>, zuku::rw_vector<T>> {
  zuku::rw_vector<T> operator()(std::vector<Store<T>>&& src) {
    zuku::rw_vector<T> dst;
    dst.reserve(src.size());
    for (auto&& element : src) {
      dst.emplace_back(ResolveDependency<Store<T>, T&>{}(element));
    }
    return dst;
  }
};

template <class T>
struct ResolveDependency<std::vector<View<T>>, zuku::ro_vector<T>> {
  zuku::ro_vector<T> operator()(std::vector<View<T>>&& src) {
    zuku::ro_vector<T> dst;
    dst.reserve(src.size());
    for (auto&& element : src) {
      dst.push_back(ResolveDependency<View<T>, const T&>{}(std::move(element)));
    }
    return dst;
  }
};

template <class T>
struct ResolveDependency<std::vector<Future<T>>, std::vector<T>> {
  std::vector<T> operator()(std::vector<Future<T>>&& src) {
    std::vector<T> dst;
    dst.reserve(src.size());
    for (auto&& element : src) {
      dst.push_back(ResolveDependency<Future<T>, T>{}(std::move(element)));
    }
    return dst;
  }
};

template <class SourceTuple, class TargetTuple, std::size_t TargetOffset,
          std::size_t SourceIndex>
decltype(auto) resolve_dependency_tuple(SourceTuple& args) {
  using source_type = std::tuple_element_t<SourceIndex, SourceTuple>;
  using target_type =
      std::tuple_element_t<SourceIndex + TargetOffset, TargetTuple>;

  if constexpr (is_store<source_type>::value && !is_store<target_type>::value) {
    return ResolveDependency<source_type, target_type>{}(
        std::get<SourceIndex>(args));
  } else {
    return ResolveDependency<source_type, target_type>{}(
        std::move(std::get<SourceIndex>(args)));
  }
}

}  // namespace zuku

#endif