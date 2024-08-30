/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef _POC_SRC_PROMISE_H_
#define _POC_SRC_PROMISE_H_

#include <stdexcept>
#include <tuple>

#include "future.h"

namespace zuku {

template <class T>
class Promise {
 public:
  Promise(Realm::UserEvent ev)
      : deferred_event_(std::move(ev)),
        value_(Future<T>::CreateEmptyShared()) {}

  Promise() : value_(std::make_shared<std::optional<T>>(std::nullopt)) {}

  void emplace(T&& t) {
    auto& ptr = std::get<std::shared_ptr<std::optional<T>>>(value_);
    *ptr = std::move(t);
  }

  bool HasDeferredEvent() const { return deferred_event_.has_value(); }

  const Realm::UserEvent DeferredEvent() const { return *deferred_event_; }

  const std::shared_ptr<Future<T>>& get_deferred_value() const {
    return std::get<std::shared_ptr<Future<T>>>(value_);
  }

  const std::shared_ptr<std::optional<T>>& Value() const {
    return std::get<std::shared_ptr<std::optional<T>>>(value_);
  }

  void emplace(std::shared_ptr<Future<T>> deferred) {
    // auto& ptr = std::get<std::shared_ptr<Future<T>>>(value_);
    value_ = std::move(deferred);
    //*ptr = *std::move(deferred);
  }

  void emplace(std::shared_ptr<std::optional<T>> value) {
    auto& ptr = std::get<std::shared_ptr<Future<T>>>(value_);
    ptr->emplace(std::move(value));
  }

 private:
  std::optional<Realm::UserEvent> deferred_event_;

  std::variant<std::shared_ptr<Future<T>>, std::shared_ptr<std::optional<T>>>
      value_;
};

template <>
class Promise<void> {
 public:
  Promise(Realm::UserEvent ev) : deferred_event_(std::move(ev)) {}

  Promise() = default;

  bool HasDeferredEvent() const { return deferred_event_.has_value(); }

  const Realm::UserEvent DeferredEvent() const { return *deferred_event_; }

 private:
  std::optional<Realm::UserEvent> deferred_event_;
};

template <class FutureType, class TaskReturnType, int Index>
struct CreatePromise;

template <class T>
auto create_concrete_promise() {
  return Promise<T>{};
}

template <class T>
auto create_deferred_promise() {
  return Promise<T>{Realm::UserEvent::create_user_event()};
}

template <class T, int Index>
struct CreatePromise<Future<T>, Future<T>, Index> {
  template <class Task>
  auto operator()(Task& task) {
    // the future will not be resolved by the task
    return create_deferred_promise<T>();
  }
};

template <class T, int Index>
struct CreatePromise<Future<T>, T, Index> {
  template <class Task>
  auto operator()(Task& task) {
    return create_concrete_promise<T>();
  }
};

template <class T, int Index>
struct CreatePromise<Future<std::vector<T>>, std::vector<T>, Index> {
  template <class Task>
  auto operator()(Task& task) {
    static_assert(task.template has_output_bouds<Index>());
    if constexpr (task.template has_output_bounds<Index>()) {
      std::vector<Future<T>> ret;
      std::size_t num_outputs = task.template output_bound<Index>();
      ret.reserve(num_outputs);
      for (std::size_t i = 0; i < num_outputs; ++i) {
        ret.push_back(CreatePromise<T, T, Index>{}(task));
      }
      return ret;
    } else {
      return create_concrete_promise<std::vector<T>>();
    }
  }
};

template <class ReturnTuple, std::size_t Index, class Task>
auto make_promise_tuple_element(Task& task) {
  using return_type = typename std::tuple_element_t<Index, ReturnTuple>;
  using future_type = typename DecayNestedFuture<return_type>::type;

  return CreatePromise<future_type, return_type, Index>{}(task);
}

template <class ReturnTuple, class Task, std::size_t... I>
auto make_promise_tuple(Task& task, std::index_sequence<I...>) {
  return std::make_tuple(make_promise_tuple_element<ReturnTuple, I>(task)...);
}

template <class T>
struct PromiseWrappedReturn {
  template <class Task>
  auto operator()(Task& task) {
    return make_promise_tuple<std::tuple<T>>(task,
                                             std::make_index_sequence<1>{});
  }
};

template <>
struct PromiseWrappedReturn<void> {
  template <class Task>
  Promise<void> operator()(Task& task) {
    return Promise<void>{};
  }
};

template <>
struct PromiseWrappedReturn<std::tuple<void>> {
  template <class Task>
  Promise<void> operator()(Task& task) {
    return Promise<void>{};
  }
};

template <typename... Args>
struct PromiseWrappedReturn<std::tuple<Args...>> {
  template <class Task>
  auto operator()(Task& task) {
    using ret_type = std::tuple<Args...>;
    return make_promise_tuple<ret_type>(
        task, std::make_index_sequence<std::tuple_size<ret_type>{}>{});
  }
};

template <class ReturnType, class PromiseType>
struct FulfillPromise;

template <class T>
struct FulfillPromise<T, Promise<T>> {
  void operator()(T&& t, Promise<T>& promise) {
    // should have a concrete value
    promise.emplace(std::move(t));
  }
};

template <class T>
struct FulfillPromise<std::vector<T>, std::vector<Promise<T>>> {
  void operator()(std::vector<T>&& t, std::vector<Promise<T>>& promise) {
    if (t.size() != promise.size()) {
      throw std::runtime_error(
          "promise vector does not match size of return vector");
    }

    for (std::size_t idx = 0; idx < t.size(); ++idx) {
      FulfillPromise<T, Promise<T>>(std::move(t[idx]), promise[idx]);
    }
  }
};

template <class T>
struct FulfillPromise<std::vector<Future<T>>, std::vector<Promise<T>>> {
  void operator()(std::vector<T>&& t, std::vector<Promise<T>>& promise) {
    if (t.size() != promise.size()) {
      throw std::runtime_error(
          "promise vector does not match size of return vector");
    }

    for (std::size_t idx = 0; idx < t.size(); ++idx) {
      FulfillPromise<T, Promise<T>>(std::move(t[idx]), promise[idx]);
    }
  }
};

template <class T>
struct FulfillPromise<Future<T>, Promise<T>> {
  void operator()(Future<T>&& f, Promise<T>& promise) {
    if (!promise.HasDeferredEvent()) {
      throw std::runtime_error(
          "fulfilling promise with future, but promise has no defer sync "
          "event");
    }

    if (!f.HasPrecondition()) {
      throw std::runtime_error(
          "fulfilling promise with future, but future has no precondition");
    }

    if (f.HasDeferredValue()) {
      promise.emplace(std::forward<Future<T>>(f).get_deferred_value());
    } else {
      promise.emplace(std::forward<Future<T>>(f).shared_value());
    }

    // the promise can trigger the user event to notify futures are ready
    // whenever the future's precondition has been satisfied
    promise.DeferredEvent().trigger(f.Precondition());
  }
};

template <>
struct FulfillPromise<Future<void>, Promise<void>> {
  void operator()(Future<void>&& f, Promise<void>& promise) {
    if (!f.HasPrecondition()) {
      throw std::runtime_error(
          "fulfilling promise with future, but future has no precondition");
    }

    // the promise can trigger the user event to notify futures are ready
    // whenever the future's precondition has been satisfied
    promise.DeferredEvent().trigger(f.Precondition());
  }
};

template <class T, std::size_t Index, std::size_t Remaining>
struct FulfillPromises {
  template <class Task>
  void operator()(Task& task, T&& result);
};

template <class... Args, std::size_t Index>
struct FulfillPromises<std::tuple<Args...>, Index, 0> {
  template <class Task>
  void operator()(Task& task, std::tuple<Args...>&& result) {}
};

template <class... Args, std::size_t Index, std::size_t Remaining>
struct FulfillPromises<std::tuple<Args...>, Index, Remaining> {
  template <class Task>
  void operator()(Task& task, std::tuple<Args...>&& result) {
    using return_t = std::tuple_element_t<Index, std::tuple<Args...>>;
    using promise_t = typename Task::template promise_element_t<Index>;
    // extract the element at index as an r-value
    FulfillPromise<return_t, promise_t>{}(
        std::get<Index>(std::forward<std::tuple<Args...>>(result)),
        task.template promise<Index>());
    FulfillPromises<std::tuple<Args...>, Index + 1, Remaining - 1>{}(
        task, std::move(result));
  }
};

template <class T, std::size_t Index, std::size_t Remaining>
template <class Task>
void FulfillPromises<T, Index, Remaining>::operator()(Task& task, T&& result) {
  FulfillPromises<std::tuple<T>, Index, Remaining>{}(
      task, std::make_tuple(std::move(result)));
}

}  // namespace zuku

#endif