/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef _POC_SRC_TYPE_TRAITS_H_
#define _POC_SRC_TYPE_TRAITS_H_

#include "realm.h"

#include <tuple>
#include <type_traits>
#include <vector>
#include <optional>

namespace zuku {

template <class Signature>
struct LambdaTraits {};

template <class Ret, class T, class... Args>
struct LambdaTraits<Ret (T::*)(Args...) const> {
  using arg_tuple = std::tuple<Args...>;
  using ret_type = Ret;
};

template <class T>
struct Function {
  using arg_tuple = typename LambdaTraits<decltype(&T::operator())>::arg_tuple;
  using ret_type = typename LambdaTraits<decltype(&T::operator())>::ret_type;
};

template <class T>
struct Tupleify {
  using type = std::tuple<T>;
};

template <>
struct Tupleify<void> {
  using type = void;
};

template <class T, class U>
struct Tupleify<std::pair<T, U>> {
  using type = std::tuple<T, U>;
};

template <class... Args>
struct Tupleify<std::tuple<Args...>> {
  using type = std::tuple<Args...>;
};

template <typename T>
struct PrintType;

template <typename T, std::size_t Index>
struct PrintTypeAndIndex;

template <class T>
struct is_tuple : public std::false_type {};

template <class... Args>
struct is_tuple<std::tuple<Args...>> : public std::true_type {};

template <typename>
struct is_vector : std::false_type {};

template <typename T, typename Allocator>
struct is_vector<std::vector<T, Allocator>> : std::true_type {};

template <class... Ts>
struct overloaded : Ts... {
  using Ts::operator()...;
};

template <class... Ts>
overloaded(Ts...) -> overloaded<Ts...>;

template <typename>
struct is_deferred : std::false_type {};

template <typename Rect>
struct rect_info {};

template <int ndims, typename T>
struct rect_info<Realm::Rect<ndims, T>> {
  static constexpr int N = ndims;
  using index_type = T;
};

template <int N, typename... Args>
struct has_deferred_element;

template <>
struct has_deferred_element<0> {
  static constexpr bool value = false;
};

template <int N, typename T, typename... Args>
struct has_deferred_element<N, T, Args...> {
  static constexpr bool value = is_deferred<std::decay_t<T>>::value ||
                                has_deferred_element<N - 1, Args...>::value;
};

template <typename T>
struct tuple_has_deferred_element;

template <typename... Args>
struct tuple_has_deferred_element<std::tuple<Args...>> {
  static constexpr bool value =
      has_deferred_element<sizeof...(Args), Args...>::value;
};

template <typename T,
          typename S = decltype(std::declval<T>().DeferredDeleteInstances(
              std::declval<const Realm::Event&>()))>
static std::true_type hasDeferredDeleteInstances(const Realm::Event&);

template <typename T>
static std::false_type hasDeferredDeleteInstances(...);

template <class T>
struct has_deferred_delete_instances {
  static constexpr bool value =
      decltype(hasDeferredDeleteInstances<T>(Realm::Event::NO_EVENT))::value;
};

}  // namespace zuku

#endif
