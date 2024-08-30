/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef _POC_SRC_FUTURE_H_
#define _POC_SRC_FUTURE_H_

#include "realm.h"
#include "realm/event.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "init.h"
#include "task_base.h"
#include "type_traits.h"

namespace zuku {

template <class T>
class View;

class ReleaseHolderBase {
 public:
  virtual ~ReleaseHolderBase() = default;

  const Realm::Event& event() const { return ev_; }

 protected:
  explicit ReleaseHolderBase(Realm::Event ev) : ev_(ev) {}

 private:
  Realm::Event ev_;
};

template <class T>
class ReleaseHolder : public ReleaseHolderBase {
 public:
  template <class U>
  explicit ReleaseHolder(U&& val, Realm::Event ev)
      : val_(std::forward<U>(val)), ReleaseHolderBase(ev) {}

  ~ReleaseHolder() override = default;

 private:
  T val_;
};

void deferred_release_shared_pointer(const void* args, size_t arglen,
                                     const void* userdata, size_t userlen,
                                     Realm::Processor p);

template <class T>
void deferred_release(T&& val, Realm::Event wait_on) {
  auto* holder =
      new ReleaseHolder<std::decay_t<T>>(std::forward<T>(val), wait_on);
  IncPendingOp();
  Processor::Util().RealmProc().spawn(
      (Realm::Processor::TaskFuncID)GlobalTaskId::RELEASE_SHARED_POINTER,
      &holder, sizeof(void*), wait_on);
}

struct EventPrecondition {
  Realm::Event stream;
  Realm::Event effects;
};

template <class T>
class Future {
 public:
  using Value =
      std::variant<std::shared_ptr<Future<T>>,
                   std::shared_ptr<std::optional<T>>, std::shared_ptr<T>>;

  Future(Realm::Event ev, std::shared_ptr<std::optional<T>> value)
      : precondition_(std::move(ev)), value_(std::move(value)) {}

  Future(Realm::Event ev, std::shared_ptr<Future<T>> value)
      : precondition_(std::move(ev)), value_(std::move(value)) {}

  static Future CreateEmpty() { return Future(); }

  // Constructor for value that exists and is properly allocated
  // but is not yet in a "ready" state
  Future(Realm::Event ev, T&& value)
      : precondition_(std::move(ev)),
        value_(std::make_shared<std::optional<T>>(std::move(value))) {}

  Future(const Future<T>& src) = delete;

  Future(Future<T>&& src) = default;

  bool HasPrecondition() const {
    return precondition_.has_value() || war_precondition_.has_value();
  }

  Realm::Event Precondition() const {
    // a WAR precondition is a stronger precondition
    // than a RAW precondition. A WAR cannot exist until
    // the original RAW has been satisfied.
    if (war_precondition_.has_value()) {
      return *war_precondition_;
    }
    if (precondition_.has_value()) {
      return *precondition_;
    }
    return Realm::Event::NO_EVENT;
  }

  Realm::Event StreamPrecondition() const { return Precondition(); }

  Realm::Event EffectsPrecondition() const { return Precondition(); }

  void emplace(std::shared_ptr<std::optional<T>> value) {
    // the value here is no longer deferred
    value_ = std::move(value);
  }

  bool HasDeferredValue() const {
    return std::holds_alternative<std::shared_ptr<Future<T>>>(value_);
  }

  std::shared_ptr<Future<T>> get_deferred_value() && {
    return std::get<std::shared_ptr<Future<T>>>(std::move(value_));
  }

  std::shared_ptr<std::optional<T>> shared_value() && {
    return std::get<std::shared_ptr<std::optional<T>>>(std::move(value_));
  }

  const std::shared_ptr<std::optional<T>>& shared_value() const& {
    return std::get<std::shared_ptr<std::optional<T>>>(value_);
  }

  const T& get_value() & {
    if (std::holds_alternative<std::shared_ptr<Future<T>>>(value_)) {
      auto& ptr = std::get<std::shared_ptr<Future<T>>>(value_);
      return ptr->get_value();
    }
    auto& ptr = std::get<std::shared_ptr<std::optional<T>>>(value_);
    return ptr->value();
  }

  T get_value() && {
    if (std::holds_alternative<std::shared_ptr<Future<T>>>(value_)) {
      auto& ptr = std::get<std::shared_ptr<Future<T>>>(value_);
      return std::move(*ptr).get_value();
    }
    auto& ptr = std::get<std::shared_ptr<std::optional<T>>>(value_);
    return *std::move(*ptr);
  }

  static std::shared_ptr<Future<T>> CreateEmptyShared() {
    return std::make_shared<Future<T>>(Future<T>{});
  }

  const T& wait_and_get() & {
    if (precondition_.has_value()) {
      precondition_->wait();
    }
    return get_value();
  }

  View<T> view();

  View<T> view(const Realm::Event& task_completion_ev);

 private:
  Future() {}

  std::optional<Realm::Event> precondition_;
  std::optional<Realm::Event> war_precondition_;

  Value value_;
};

template <class T>
class View {
 public:
  View(std::optional<EventPrecondition> precondition,
       std::optional<Realm::UserEvent> war_dependency,
       typename Future<T>::Value value)
      : precondition_(std::move(precondition)),
        war_dependency_(std::move(war_dependency)),
        value_(std::move(value)) {}

  View(const View&) = delete;

  View(View&& v) {
    precondition_ = std::move(v.precondition_);
    war_dependency_ = std::move(v.war_dependency_);
    value_ = std::move(v.value_);
    v.war_dependency_ = std::nullopt;
  }

  ~View() {
    // if moved out of, this might be empty
    if (war_dependency_.has_value()) {
      Realm::Event subview_release_event = [&] {
        if (views_.empty()) {
          return Realm::Event::NO_EVENT;
        }
        return Realm::Event::merge_events(views_);
      }();

      war_dependency_->trigger(subview_release_event);
      if (!views_.empty()) {
        std::visit(overloaded{[&](auto ptr) {
                     deferred_release(std::move(ptr), subview_release_event);
                   }},
                   std::move(value_));
      }
    }
  }

  bool HasPrecondition() const { return precondition_.has_value(); }

  Realm::Event Precondition() const {
    if (precondition_.has_value()) {
      return precondition_->effects;
    }
    return Realm::Event::NO_EVENT;
  }

  Realm::Event StreamPrecondition() const {
    if (precondition_.has_value()) {
      return precondition_->stream;
    }
    return Realm::Event::NO_EVENT;
  }

  Realm::Event EffectsPrecondition() const { return Precondition(); }

  const T* operator->() const { return &(value()); }

  const T& operator*() const { return value(); }

  const T& value() const {
    if (std::holds_alternative<std::shared_ptr<Future<T>>>(value_)) {
      throw std::runtime_error(
          "cannot get value from View with unresolved Future");
    }
    if (std::holds_alternative<std::shared_ptr<T>>(value_)) {
      return *std::get<std::shared_ptr<T>>(value_);
    }
    return std::get<std::shared_ptr<std::optional<T>>>(value_)->value();
  }

  View<T> view() {
    // the original war_dependency cannot be triggered
    // until all subviews have been released
    Realm::UserEvent view_release_event = Realm::UserEvent::create_user_event();
    views_.emplace_back(view_release_event);
    return View<T>(precondition_, std::move(view_release_event), value_);
  }

  View<T> view(const Realm::Event& task_completion_ev) {
    views_.emplace_back(task_completion_ev);
    return View<T>(precondition_, std::nullopt, value_);
  }

  void DoneAfter(Realm::Event ev) {
    // treat this as-if it is an extra view
    views_.push_back(std::move(ev));
  }

 private:
  std::optional<EventPrecondition> precondition_;
  std::optional<Realm::UserEvent> war_dependency_;
  std::vector<Realm::Event> views_;
  typename Future<T>::Value value_;
};

template <typename T>
struct is_deferred<View<T>> : std::true_type {};

template <class T>
View<T> Future<T>::view() {
  Realm::UserEvent reader = Realm::UserEvent::create_user_event();
  if (war_precondition_.has_value()) {
    war_precondition_ = Realm::Event::merge_events(*war_precondition_, reader);
  } else {
    war_precondition_ = reader;
  }

  auto view_precondition = [&]() -> std::optional<EventPrecondition> {
    if (precondition_.has_value()) {
      return EventPrecondition{.stream = *precondition_,
                               .effects = *precondition_};
    }
    return std::nullopt;
  }();

  return View<T>(std::move(view_precondition), std::move(reader), value_);
}

template <class T>
View<T> Future<T>::view(const Realm::Event& task_completion_ev) {
  if (war_precondition_.has_value()) {
    war_precondition_ =
        Realm::Event::merge_events(*war_precondition_, task_completion_ev);
  } else {
    war_precondition_ = task_completion_ev;
  }

  auto view_precondition = [&]() -> std::optional<EventPrecondition> {
    if (precondition_.has_value()) {
      return EventPrecondition{.stream = *precondition_,
                               .effects = *precondition_};
    }
    return std::nullopt;
  }();

  return View<T>(std::move(view_precondition), std::nullopt, value_);
}

template <>
class Future<void> {
 public:
  bool HasPrecondition() const { return precondition_.has_value(); }

  static constexpr bool HasDeferredValue() { return false; }

  const Realm::Event& Precondition() const { return *precondition_; }

  explicit Future(Realm::Event ev) : precondition_(std::move(ev)) {}

 private:
  std::optional<Realm::Event> precondition_;
};

template <class T>
struct is_future : public std::false_type {};

template <class T>
struct is_future<Future<T>> : public std::true_type {};

template <typename T>
struct is_deferred<Future<T>> : std::true_type {};

template <class T>
struct DecayNestedFuture {
  using type = Future<T>;
};

template <class T>
struct DecayNestedFuture<Future<Future<T>>> {
  using type = Future<T>;
};

template <class T>
struct DecayNestedFuture<Future<T>> {
  using type = Future<T>;
};

template <class T, class Promise>
auto create_concrete_future(const Promise& promise, const Realm::Event& ev) {
  return Future<T>{ev, promise.Value()};
}

template <class T, class Promise>
auto create_deferred_future(const Promise& promise) {
  if constexpr (std::is_void_v<T>) {
    return Future<void>(promise.DeferredEvent());
  } else {
    return Future<T>{promise.DeferredEvent(), promise.get_deferred_value()};
  }
}

template <class FutureType, class TaskReturnType, int Index>
struct CreateFuture;

template <class T, int Index>
struct CreateFuture<Future<T>, Future<T>, Index> {
  template <class Task, class Promise>
  auto operator()(Task& task, const Promise& promise, const Realm::Event& ev) {
    // the future will not be resolved by the task
    return create_deferred_future<T>(promise);
  }
};

template <class T, int Index>
struct CreateFuture<Future<T>, T, Index> {
  template <class Task, class Promise>
  auto operator()(Task& task, const Promise& promise, const Realm::Event& ev) {
    return create_concrete_future<T>(promise, ev);
  }
};

template <class T, int Index>
struct CreateFuture<Future<std::vector<T>>, std::vector<T>, Index> {
  template <class Task, class Promise>
  auto operator()(Task& task, const Promise& promise, const Realm::Event& ev) {
    static_assert(task.template has_output_bouds<Index>());
    if constexpr (task.template has_output_bounds<Index>()) {
      std::vector<Future<T>> ret;
      std::size_t num_outputs = task.template output_bound<Index>();
      ret.reserve(num_outputs);
      for (std::size_t i = 0; i < num_outputs; ++i) {
        ret.push_back(CreateFuture<T, T, Index>{}(task, promise[i], ev));
      }
      return ret;
    } else {
      return create_concrete_future<std::vector<T>, Index>(task, promise, ev);
    }
  }
};

template <class ReturnTuple, std::size_t Index, class Task>
auto make_future_tuple_element(Task& task, const Realm::Event& ev) {
  using return_type = typename std::tuple_element_t<Index, ReturnTuple>;
  using future_type = typename DecayNestedFuture<return_type>::type;

  return CreateFuture<future_type, return_type, Index>{}(
      task, task.template promise<Index>(), ev);
}

template <class ReturnTuple, class Task, std::size_t... I>
auto make_future_tuple(Task& task, const Realm::Event& ev,
                       std::index_sequence<I...>) {
  return std::make_tuple(
      make_future_tuple_element<ReturnTuple, I>(task, ev)...);
};

template <class T>
struct FutureWrappedReturn {
  template <class Task>
  auto operator()(Task& task) {
    return make_future_tuple<std::tuple<T>>(task,
                                            std::make_index_sequence<1>{});
  }
};

template <>
struct FutureWrappedReturn<void> {
  auto operator()(TaskBase& task, Realm::Event ev) {
    return std::make_tuple(Future<void>{std::move(ev)});
  }
};

template <>
struct FutureWrappedReturn<std::tuple<void>> {
  auto operator()(TaskBase& task, Realm::Event ev) {
    return std::make_tuple(Future<void>{std::move(ev)});
  }
};

template <typename... Args>
struct FutureWrappedReturn<std::tuple<Args...>> {
  template <class Task>
  auto operator()(Task& task, Realm::Event ev) {
    using ret_type = std::tuple<Args...>;
    return make_future_tuple<ret_type>(
        task, ev, std::make_index_sequence<std::tuple_size_v<ret_type>>{});
  }
};

template <typename T>
struct ToTaskArgument<Future<T>&, Future<T>> {
  auto operator()(const Realm::Event& stream, const Realm::Event& effects,
                  Future<T>& f) const {
    // mapping a future to a future that might create more tasks
    // create a view with an open-ended finish event
    return f.view();
  }
};

template <typename T>
struct ToTaskArgument<Future<T>&, T&> {
  auto operator()(const Realm::Event& stream, const Realm::Event& effects,
                  Future<T>& f) const {
    // static_assert(false, "cannot map future to reference argument");
    return f.view(stream, effects);
  }
};

template <typename T>
struct ToTaskArgument<Future<T>&, const T&> {
  auto operator()(const Realm::Event& stream, const Realm::Event& effects,
                  Future<T>& f) const {
    // mapping a future to a read-only argument
    // the will not be able to create subviews
    return f.view(stream, effects);
  }
};

template <typename T>
struct ToTaskArgument<Future<T>&, T> {
  auto operator()(const Realm::Event& stream, const Realm::Event& effects,
                  Future<T>& f) const {
    // mapping a future to a read-only argument
    // thhat will not be able to create subviews
    return f.view(effects);
  }
};

template <typename T>
struct ToTaskArgument<View<T>&, View<T>> {
  auto operator()(const Realm::Event& stream, const Realm::Event& effects,
                  View<T>& v) const {
    // mapping a view to a future view might create more tasks
    // create a view with an open-ended finish event
    return v.view();
  }
};

template <typename T>
struct ToTaskArgument<View<T>&, const T&> {
  auto operator()(const Realm::Event& stream, const Realm::Event& effects,
                  View<T>& f) const {
    // mapping a view to a read-only argument
    // the will not be able to create subviews
    return f.view(stream, effects);
  }
};

template <typename T>
struct ToTaskArgument<View<T>&, T> {
  auto operator()(const Realm::Event& stream, const Realm::Event& effects,
                  View<T>& f) const {
    // mapping a view to a read-only argument
    // thhat will not be able to create subviews
    return f.view(stream, effects);
  }
};

}  // namespace zuku

#endif
