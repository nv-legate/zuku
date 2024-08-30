/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef _POC_SRC_TOKEN_H_
#define _POC_SRC_TOKEN_H_

#include "realm.h"

#include <tuple>

#include "future.h"
#include "promise.h"

template <class ReturnTuple>
class ReturnToken {
 public:
  ReturnToken(Realm::Event ev, ReturnTuple&& t,
              std::optional<Realm::UserEvent> control_event = std::nullopt)
      : ev_(std::move(ev)),
        return_(std::move(t)),
        control_event_(std::move(control_event)) {}

  template <std::size_t Index>
  std::tuple_element_t<Index, ReturnToken>& get() & {
    return std::get<Index>(return_);
  }

  const Realm::Event& Event() const { return ev_; }

  bool HasControlEvent() const { return control_event_.has_value(); }

  const Realm::UserEvent& ControlEvent() const { return *control_event_; }

  ReturnTuple&& futures() && { return std::move(return_); }

  template <std::size_t Index>
  std::tuple_element_t<Index, ReturnToken> const& get() const& {
    return std::get<Index>(return_);
  }

  template <std::size_t Index>
  std::tuple_element_t<Index, ReturnToken>& get() && {
    return std::get<Index>(return_);
  }

  template <std::size_t Index>
  std::tuple_element_t<Index, ReturnToken> const& get() const&& {
    return std::get<Index>(return_);
  }

  void Wait() { ev_.wait(); }

 private:
  Realm::Event ev_;
  ReturnTuple return_;
  std::optional<Realm::UserEvent> control_event_;
};

namespace std {
template <class ReturnTuple>
struct tuple_size<ReturnToken<ReturnTuple>>
    : std::integral_constant<size_t, std::tuple_size<ReturnTuple>{}> {};

template <size_t Index, class ReturnTuple>
struct tuple_element<Index, ReturnToken<ReturnTuple>> {
  using type = tuple_element_t<Index, ReturnTuple>;
};

}  // namespace std

namespace zuku {

template <class... Args, int Index>
struct CreateFuture<Future<ReturnToken<std::tuple<Args...>>>,
                    ReturnToken<std::tuple<Args...>>, Index> {
  template <class Task, class Promise>
  auto operator()(Task& task, const Promise& promise, const Realm::Event& ev) {
    // the future will not be resolved by the task
    return create_deferred_future<void>(promise);
  }
};

template <class... Args, int Index>
struct CreatePromise<Future<ReturnToken<std::tuple<Args...>>>,
                     ReturnToken<std::tuple<Args...>>, Index> {
  template <class Task>
  auto operator()(Task& task) {
    // the future will not be resolved by the task
    return create_deferred_promise<void>();
  }
};

template <>
struct FulfillPromise<ReturnToken<std::tuple<>>, Promise<void>> {
  void operator()(ReturnToken<std::tuple<>>&& f, Promise<void>& promise) {
    // the promise can trigger the user event to notify futures are ready
    // whenever the future's precondition has been satisfied
    promise.DeferredEvent().trigger(f.Event());
  }
};

template <>
struct FulfillPromise<ReturnToken<std::tuple<Future<void>>>, Promise<void>> {
  void operator()(ReturnToken<std::tuple<Future<void>>>&& f,
                  Promise<void>& promise) {
    // the promise can trigger the user event to notify futures are ready
    // whenever the future's precondition has been satisfied
    promise.DeferredEvent().trigger(f.Event());
  }
};

}  // namespace zuku

#endif