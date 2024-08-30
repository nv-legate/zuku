/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef _POC_SRC_STORE_H_
#define _POC_SRC_STORE_H_

#include "realm/event.h"

#include <functional>
#include <optional>
#include <vector>

#include "future.h"
#include "type_traits.h"
#include "vector.h"

namespace zuku {

template <class T>
class Store {
 public:
  struct Ctor {
    bool has_parent{false};
    std::optional<Realm::UserEvent> parent_notificaton_event{std::nullopt};
    std::optional<EventPrecondition> child_rw_event{std::nullopt};
    std::optional<EventPrecondition> war_precondition{std::nullopt};
    std::shared_ptr<T> value{};
  };

  template <class... Args>
  static Store<T> Create(Args&&... args) {
    return Store({.value = std::make_shared<T>(std::forward<Args>(args)...)});
  }

  static Store CreateEmpty() { return Store{{}}; }

  template <class... Args>
  static Store<T> DeferredCreate(Realm::Event create_event, Args&&... args) {
    EventPrecondition precondition = {.stream = create_event,
                                      .effects = create_event};
    return Store({.child_rw_event = std::move(precondition),
                  .value = std::make_shared<T>(std::forward<Args>(args)...)});
  }

  Store(const Store<T>& src) = delete;

  Store(Store<T>&& v) {
    parent_notification_event_ = std::move(v.parent_notification_event_);
    war_precondition_ = std::move(v.war_precondition_);
    child_rw_event_ = std::move(v.child_rw_event_);
    value_ = std::move(v.value_);
    has_parent_ = v.has_parent_;
    v.value_ = nullptr;
    v.parent_notification_event_ = std::nullopt;
    v.child_rw_event_ = std::nullopt;
  }

  explicit Store(Ctor&& ctor_args)
      : has_parent_(ctor_args.has_parent),
        parent_notification_event_(
            std::move(ctor_args.parent_notificaton_event)),
        child_rw_event_(std::move(ctor_args.child_rw_event)),
        war_precondition_(std::move(ctor_args.war_precondition)),
        value_(std::move(ctor_args.value)) {}

  ~Store() {
    if (parent_notification_event_.has_value()) {
      Realm::Event child_event = child_rw_event_.has_value()
                                     ? child_rw_event_->effects
                                     : Realm::Event::NO_EVENT;
      parent_notification_event_->trigger(child_event);
    }

    if constexpr (has_deferred_delete_instances<T>::value) {
      if (!has_parent_ && value_ && HasPrecondition()) {
        value_->DeferredDeleteInstances(Precondition());
      }
    }
    if (child_rw_event_.has_value() && value_) {
      // we can't throw away the store until we are sure all pending ops on
      // this have finished
      deferred_release(std::move(value_), child_rw_event_->effects);
    }
  }

  Store& operator=(Store&& v) {
    parent_notification_event_ = std::move(v.parent_notification_event_);
    war_precondition_ = std::move(v.war_precondition_);
    child_rw_event_ = std::move(v.child_rw_event_);
    value_ = std::move(v.value_);
    has_parent_ = v.has_parent_;
    v.value_ = nullptr;
    v.parent_notification_event_ = std::nullopt;
    v.child_rw_event_ = std::nullopt;
    return *this;
  }

  void Wait() {
    if (child_rw_event_.has_value()) {
      child_rw_event_->effects.wait();
      child_rw_event_ = std::nullopt;
    }
    if (war_precondition_.has_value()) {
      war_precondition_->effects.wait();
      war_precondition_ = std::nullopt;
    }
  }

  bool HasWarPrecondition() const { return war_precondition_.has_value(); }

  bool HasPrecondition() const {
    return war_precondition_.has_value() || child_rw_event_.has_value();
  }

  bool HasPostcondition() const {
    return parent_notification_event_.has_value();
  }

  Realm::Event Precondition() const {
    if (war_precondition_.has_value()) {
      if (child_rw_event_.has_value()) {
        return Realm::Event::merge_events(war_precondition_->effects,
                                          child_rw_event_->effects);
      }
      return war_precondition_->effects;
    }

    if (child_rw_event_.has_value()) {
      return child_rw_event_->effects;
    }
    return Realm::Event::NO_EVENT;
  }

  Realm::Event StreamPrecondition() const {
    if (war_precondition_.has_value()) {
      if (child_rw_event_.has_value()) {
        return Realm::Event::merge_events(war_precondition_->stream,
                                          child_rw_event_->stream);
      }
      return war_precondition_->stream;
    }
    if (child_rw_event_.has_value()) {
      return child_rw_event_->stream;
    }
    return Realm::Event::NO_EVENT;
  }

  Realm::Event EffectsPrecondition() const { return Precondition(); }

  Realm::Event Postcondition() const {
    if (parent_notification_event_.has_value()) {
      return *parent_notification_event_;
    }
    return Realm::Event::NO_EVENT;
  }

  Store<T> child(const Realm::Event& stream, const Realm::Event& effects) {
    // there is no parent notification event
    // this child will be fully actualized
    // after the event completes and the child
    // cannot be used to create more child stores
    Store<T> new_child({.has_parent = true,
                        .parent_notificaton_event = std::nullopt,
                        .child_rw_event = std::move(child_rw_event_),
                        .war_precondition = war_precondition_,
                        .value = value_});

    ReadyAfter(stream, effects);
    return new_child;
  }

  Store<T> child(const Realm::Event& ev) { return child(ev, ev); }

  Store<T> child() {
    // merge any existing child_rw_event with this one
    Realm::UserEvent new_child_event = Realm::UserEvent::create_user_event();

    Store<T> new_child({.has_parent = true,
                        .parent_notificaton_event = std::move(new_child_event),
                        .child_rw_event = std::move(child_rw_event_),
                        .war_precondition = war_precondition_,
                        .value = value_});

    ReadyAfter(new_child_event);

    return new_child;
  }

  void ReadyAfter(Realm::Event stream, Realm::Event effects) {
    if (child_rw_event_.has_value()) {
      child_rw_event_->effects =
          Realm::Event::merge_events(effects, child_rw_event_->effects);
      child_rw_event_->stream =
          Realm::Event::merge_events(stream, child_rw_event_->stream);
    } else {
      child_rw_event_ = EventPrecondition{.stream = stream, .effects = effects};
    }
    // this is no longer relevant, only the child_rw_event is a precondition
    war_precondition_ = std::nullopt;
  }

  void ReadyAfter(Realm::Event ev) { ReadyAfter(ev, ev); }

  const T& shared_value() const& { return *value_; }

  T& get_value() & { return *value_; }

  const T& get_value() const& { return *value_; }

  T& operator*() & { return *value_; }

  T&& operator*() && { return std::move(*value_); }

  const T& operator*() const& { return *value_; }

  T* operator->() & { return &(get_value()); }

  const T* operator->() const& { return &(get_value()); }

  View<T> unsafe_view() {
    return View<T>(child_rw_event_, std::nullopt, value_);
  }

  View<T> view(const Realm::Event& stream, const Realm::Event& effects) {
    if (war_precondition_.has_value()) {
      war_precondition_->effects =
          Realm::Event::merge_events(war_precondition_->effects, effects);
      war_precondition_->stream =
          Realm::Event::merge_events(war_precondition_->stream, stream);
    } else {
      war_precondition_ =
          EventPrecondition{.stream = stream, .effects = effects};
    }
    return View<T>(child_rw_event_, std::nullopt, value_);
  }

  View<T> view(const Realm::Event& ev) { return view(ev, ev); }

  View<T> view() {
    Realm::UserEvent reader = Realm::UserEvent::create_user_event();
    if (war_precondition_.has_value()) {
      war_precondition_->effects =
          Realm::Event::merge_events(war_precondition_->effects, reader);
      war_precondition_->stream =
          Realm::Event::merge_events(war_precondition_->stream, reader);
    } else {
      war_precondition_ =
          EventPrecondition{.stream = reader, .effects = reader};
    }
    return View<T>(child_rw_event_, std::move(reader), value_);
  }

 private:
  bool has_parent_;
  std::optional<Realm::UserEvent> parent_notification_event_;
  std::optional<EventPrecondition> child_rw_event_;
  std::optional<EventPrecondition> war_precondition_;
  std::shared_ptr<T> value_;
};  // namespace zuku

template <typename>
struct is_store : std::false_type {};

template <typename T>
struct is_store<Store<T>> : std::true_type {};

template <typename T>
struct is_deferred<Store<T>> : std::true_type {};

template <typename T>
struct ToTaskArgument<Store<T>&, T&> {
  auto operator()(const Realm::Event& stream, const Realm::Event& effects,
                  Store<T>& f) const {
    return f.child(stream, effects);
  }
};

template <typename T>
struct ToTaskArgument<Store<T>&, T> {
  auto operator()(const Realm::Event& stream, const Realm::Event& effects,
                  Store<T>& f) const {
    return f.view(stream, effects);
  }
};

template <typename T>
struct ToTaskArgument<Store<T>&, const T&> {
  auto operator()(const Realm::Event& stream, const Realm::Event& effects,
                  Store<T>& f) const {
    return f.view(stream, effects);
  }
};

template <typename T>
struct ToTaskArgument<Store<T>&, Store<T>> {
  auto operator()(const Realm::Event& stream, const Realm::Event& effects,
                  Store<T>& f) const {
    // create an open-ended child
    // the parent will not be declared ready until
    // child is deleted and all sub-children are also deleted
    return f.child();
  }
};

template <typename T>
struct ToTaskArgument<Store<T>&, View<T>> {
  auto operator()(const Realm::Event& stream, const Realm::Event& effects,
                  Store<T>& f) const {
    // create an open-ended child
    // the parent will not be declared ready until
    // child is deleted and all sub-children are also deleted
    return f.view();
  }
};

template <typename T>
struct ToTaskArgument<std::vector<Store<T>>&, zuku::rw_vector<T>> {
  auto operator()(const Realm::Event& stream, const Realm::Event& effects,
                  std::vector<Store<T>>& stores) {
    // we are mapping a vector of stores into a vector of references
    // which means that none of the stores can create further children
    // in subtasks
    std::vector<Store<T>> task_stored_arg;
    task_stored_arg.reserve(stores.size());
    for (Store<T>& store : stores) {
      task_stored_arg.push_back(store.child(stream, effects));
    }
    return task_stored_arg;
  }
};

template <typename T>
struct ToTaskArgument<const std::vector<Store<T>>&, ro_vector<T>> {
  auto operator()(const Realm::Event& stream, const Realm::Event& effects,
                  const std::vector<Store<T>>& stores) {
    // we are mapping a vector of stores into a vector of references
    // which means that none of the stores can create further children
    // in subtasks. the values are read-only, which means that
    // that we need views, not store children
    std::vector<View<T>> task_stored_arg;
    task_stored_arg.reserve(stores.size());
    for (Store<T>& store : stores) {
      task_stored_arg.push_back(store.view(stream, effects));
    }
    return task_stored_arg;
  }
};

template <typename T>
struct ToTaskArgument<std::vector<Store<T>>&, ro_vector<T>> {
  auto operator()(const Realm::Event& stream, const Realm::Event& effects,
                  std::vector<Store<T>>& stores) {
    return ToTaskArgument<const std::vector<T>&, zuku::ro_vector<T>>{}(
        stream, effects, stores);
  }
};

template <typename T>
struct ToTaskArgument<ro_vector<Store<T>>&, ro_vector<T>> {
  auto operator()(const Realm::Event& stream, const Realm::Event& effects,
                  ro_vector<Store<T>>& stores) {
    // none of the values in the vector can create child stores
    // so we can take a shortcut path to creating the views
    // that only depends on the task completion event ev
    std::vector<View<T>> task_stored_arg;
    task_stored_arg.reserve(stores.size());
    for (Store<T>& store : stores) {
      task_stored_arg.push_back(store.view(stream, effects));
    }
    return task_stored_arg;
  }
};

template <typename T>
struct ToTaskArgument<ro_vector<Store<T>>, ro_vector<T>> {
  auto operator()(const Realm::Event& stream, const Realm::Event& effects,
                  rw_vector<Store<T>>&& stores) {
    // none of the values in the vector can create child stores
    // so we can take a shortcut path to creating the views
    // that only depends on the task completion event ev
    std::vector<Store<T>> task_stored_arg;
    task_stored_arg.reserve(stores.size());
    for (Store<T>& store : stores) {
      task_stored_arg.push_back(store.child(stream, effects));
    }
    return task_stored_arg;
  }
};

template <typename T>
struct ToTaskArgument<rw_vector<Store<T>>&, ro_vector<T>> {
  auto operator()(const Realm::Event& stream, const Realm::Event& effects,
                  rw_vector<Store<T>>& stores) {
    // none of the values in the vector can create child stores
    // so we can take a shortcut path to creating the views
    // that only depends on the task completion event ev
    std::vector<View<T>> task_stored_arg;
    task_stored_arg.reserve(stores.size());
    for (Store<T>& store : stores) {
      task_stored_arg.push_back(store.view(stream, effects));
    }
    return task_stored_arg;
  }
};

template <typename T>
struct ToTaskArgument<rw_vector<Store<T>>, ro_vector<T>> {
  auto operator()(const Realm::Event& stream, const Realm::Event& effects,
                  rw_vector<Store<T>>&& stores) {
    // none of the values in the vector can create child stores
    // so we can take a shortcut path to creating the views
    // that only depends on the task completion event ev
    std::vector<View<T>> task_stored_arg;
    task_stored_arg.reserve(stores.size());
    for (Store<T>& store : stores) {
      task_stored_arg.push_back(store.view(stream, effects));
    }
    return task_stored_arg;
  }
};

template <typename T>
struct ToTaskArgument<rw_vector<Store<T>>&, rw_vector<T>> {
  auto operator()(const Realm::Event& stream, const Realm::Event& effects,
                  rw_vector<Store<T>>& stores) {
    // none of the values in the vector can create child stores
    // so we can take a shortcut path to creating the views
    // that only depends on the task completion event ev
    std::vector<Store<T>> task_stored_arg;
    task_stored_arg.reserve(stores.size());
    for (Store<T>& store : stores) {
      task_stored_arg.push_back(store.child(stream, effects));
    }
    return task_stored_arg;
  }
};

template <typename T>
struct ToTaskArgument<rw_vector<Store<T>>, rw_vector<T>> {
  auto operator()(const Realm::Event& stream, const Realm::Event& effects,
                  rw_vector<Store<T>>&& stores) {
    // none of the values in the vector can create child stores
    // so we can take a shortcut path to creating the views
    // that only depends on the task completion event ev
    std::vector<Store<T>> task_stored_arg;
    task_stored_arg.reserve(stores.size());
    for (Store<T>& store : stores) {
      task_stored_arg.push_back(store.child(stream, effects));
    }
    return task_stored_arg;
  }
};

template <typename T>
struct ToTaskArgument<std::vector<Store<T>>&, const zuku::ro_vector<T>&> {
  auto operator()(const Realm::Event& stream, const Realm::Event& effects,
                  std::vector<Store<T>>& stores) {
    return ToTaskArgument<const std::vector<T>&, zuku::ro_vector<T>>{}(
        stream, effects, stores);
  }
};

template <typename T>
using store_vector = rw_vector<Store<T>>;

template <class T>
class store_variant_vector {
 public:
  using value_type =
      std::variant<std::reference_wrapper<Store<T>>,
                   std::reference_wrapper<View<T>>, View<T>, Store<T>>;

  auto begin() { return values_.begin(); }

  auto end() { return values_.end(); }

  std::size_t size() const { return values_.size(); }

  void reserve(std::size_t n) { values_.reserve(n); }

  void push_back(Store<T>& store) {
    values_.emplace_back(
        std::in_place_type_t<std::reference_wrapper<Store<T>>>{}, store);
  }

  void push_back(View<T>& view) {
    values_.emplace_back(
        std::in_place_type_t<std::reference_wrapper<View<T>>>{}, view);
  }

  void push_back(Store<T>&& store) {
    values_.emplace_back(std::in_place_type_t<Store<T>>{}, std::move(store));
  }

  void push_back(View<T>&& view) {
    values_.emplace_back(std::in_place_type_t<View<T>>{}, std::move(view));
  }

  value_type& operator[](std::size_t idx) { return values_[idx]; }

 private:
  std::vector<value_type> values_;
};

// store variant can only be mapped to a read-only vector
// the inputs might be views, which are read-only
template <typename T>
struct ToTaskArgument<store_variant_vector<T>, zuku::ro_vector<T>> {
  auto operator()(const Realm::Event& stream, const Realm::Event& effects,
                  store_variant_vector<T>&& stores) {
    // none of the values in the vector can create child stores
    // so we can take a shortcut path to creating the views
    // that only depends on the task completion event ev
    std::vector<View<T>> task_stored_arg;
    task_stored_arg.reserve(stores.size());
    for (auto&& var : stores) {
      std::visit(
          overloaded{
              [&](std::reference_wrapper<Store<T>>& store_ref) {
                task_stored_arg.push_back(
                    store_ref.get().view(stream, effects));
              },
              [&](std::reference_wrapper<View<T>>& view_ref) {
                task_stored_arg.push_back(view_ref.get().view(effects));
              },
              [&](View<T>& v) { task_stored_arg.push_back(std::move(v)); },
              [&](Store<T>& s) {
                task_stored_arg.push_back(s.view(stream, effects));
              },
          },
          var);
    }
    return task_stored_arg;
  }
};

}  // namespace zuku

#endif
