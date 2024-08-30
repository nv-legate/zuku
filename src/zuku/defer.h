/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef _POC_SRC_DEFER_H_
#define _POC_SRC_DEFER_H_

#include "realm/event.h"

#include <cstdint>
#include <tuple>
#include <type_traits>
#include <variant>

#include "barrier.h"
#include "future.h"
#include "init.h"
#include "link_promise.h"
#include "mesh.h"
#include "processor.h"
#include "promise.h"
#include "realm.h"
#include "task.h"
#include "token.h"
#include "type_traits.h"

namespace zuku {

struct NoContext {
  template <class T>
  static constexpr bool has_value() {
    return false;
  }
};

struct Priority {
  int value;
};

struct NoField {};

template <class T>
void AddRealmEventPostcondition(std::set<Realm::Event>& events, const T& t) {}

template <class T>
void AddRealmEventPostcondition(std::set<Realm::Event>& events,
                                const Store<T>& s) {
  if (s.HasPrecondition()) {
    events.insert(s.Precondition());
  }
}

template <class T>
void AddRealmEventPrecondition(std::set<Realm::Event>& events, const T& t) {}

template <class T>
void AddRealmEventPrecondition(std::set<Realm::Event>& events,
                               const Future<T>& f) {
  if (f.HasPrecondition()) {
    events.insert(f.Precondition());
  }
}

template <class T>
void AddRealmEventPrecondition(std::set<Realm::Event>& events,
                               const Store<T>& s) {
  if (s.HasPrecondition()) {
    events.insert(s.Precondition());
  }
}

template <class T>
void AddRealmEventPrecondition(std::set<Realm::Event>& events,
                               const View<T>& v) {
  if (v.HasPrecondition()) {
    events.insert(v.Precondition());
  }
}

template <class T>
void AddRealmEventPrecondition(std::set<Realm::Event>& events,
                               const ReturnToken<T>& t) {
  events.insert(t.Event());
}

inline void AddRealmEventPrecondition(std::set<Realm::Event>& events,
                                      Realm::Event ev) {
  events.insert(std::move(ev));
}

inline void AddRealmEventPrecondition(std::set<Realm::Event>& events,
                                      Realm::UserEvent ev) {
  events.insert(std::move(ev));
}

template <typename Field = NoField, typename PrevContext = NoContext>
class Context {
 private:
  Field field;
  PrevContext prev_context;

 public:
  Context(Field f, PrevContext ctx)
      : field(std::move(f)), prev_context(std::move(ctx)) {}

  static Context<NoField, NoContext> Create() {
    return Context<>{NoField{}, NoContext{}};
  }

  template <typename Fxn, typename... Args>
  auto defer(Fxn f, Args&&... args);

  template <typename Arg>
  auto release(Arg&& arg) {
    // hold the value in a do-nothing Lambda
    return defer([](Arg) {}, std::move(arg));
  }

  auto add() && { return *std::move(this); }

  template <class T, class... Args>
  auto add(T&& t, Args&&... args) && {
    return Context<T, Context<Field, PrevContext>>{std::forward<T>(t),
                                                   *std::move(this)}
        .add(std::forward<Args>(args)...);
  }

  template <class T>
  auto drop() & {
    if constexpr (std::is_same_v<Field, NoField>) {
      return Context<NoField, NoContext>{NoField{}, NoContext{}};
    } else if constexpr (std::is_same_v<Field, T>) {
      return prev_context.template drop<T>();
    } else {
      return prev_context.template drop<T>(field);
    }
  };

  template <class T>
  auto drop() && {
    if constexpr (std::is_same_v<Field, NoField>) {
      return Context<NoField, NoContext>{{}, {}};
    } else if constexpr (std::is_same_v<Field, T>) {
      return std::move(prev_context).template drop<T>();
    } else {
      return std::move(prev_context).template drop<T>(std::move(field));
    }
  };

  template <class T, class FirstKeptArg, class... KeptArgs>
  auto drop(FirstKeptArg&& first, KeptArgs&&... kept_args) & {
    if constexpr (std::is_same_v<T, Field>) {
      return prev_context.template drop<T>(
          std::forward<FirstKeptArg>(first),
          std::forward<KeptArgs>(kept_args)...);
    } else if constexpr (std::is_same_v<Field, NoField>) {
      // the end, we can rebuild the context now
      return Context<NoField, NoContext>{{}, {}}.add(
          std::forward<FirstKeptArg>(first),
          std::forward<KeptArgs>(kept_args)...);
    } else {
      return prev_context.template drop<T>(
          field, std::forward<FirstKeptArg>(first),
          std::forward<KeptArgs>(kept_args)...);
    }
  }

  template <class T, class FirstKeptArg, class... KeptArgs>
  auto drop(FirstKeptArg&& first, KeptArgs&&... kept_args) && {
    if constexpr (std::is_same_v<T, Field>) {
      return std::move(prev_context)
          .template drop<T>(std::forward<FirstKeptArg>(first),
                            std::forward<KeptArgs>(kept_args)...);
    } else if constexpr (std::is_same_v<Field, NoField>) {
      // the end, we can rebuild the context now
      return Context<NoField, NoContext>{{}, {}}.add(
          std::forward<FirstKeptArg>(first),
          std::forward<KeptArgs>(kept_args)...);
    } else {
      return std::move(prev_context)
          .template drop<T>(std::move(field), std::forward<FirstKeptArg>(first),
                            std::forward<KeptArgs>(kept_args)...);
    }
  }

  template <class T>
  static constexpr bool has_value() {
    if constexpr (std::is_same_v<T, Field>) {
      return true;
    } else if constexpr (std::is_same_v<Field, NoField>) {
      return false;
    } else {
      return PrevContext::template has_value<T>();
    }
  };

  template <typename T, typename OtherContext>
  auto conditional_move_from(OtherContext& context) && {
    if constexpr (OtherContext::template has_value<T>()) {
      return std::move(*this).template add<T>(
          std::move(context.template value<T>()));
    } else {
      return std::move(*this);
    }
  }

  template <class T>
  T& value() {
    if constexpr (std::is_same_v<T, Field>) {
      return field;
    } else {
      return prev_context.template value<T>();
    }
  }

  auto if_on(Processor p) && {
    return Context<Processor, Context<Field, PrevContext>>{std::move(p),
                                                           std::move(*this)};
  }

  auto on(Processor p) && {
    return Context<Processor, Context<Field, PrevContext>>{std::move(p),
                                                           std::move(*this)};
  }

  auto on(ProcessorGroup p) && {
    return Context<ProcessorGroup, Context<Field, PrevContext>>{
        std::move(p), std::move(*this)};
  }

  auto split_control_execution() {
    return Context<ReturnControlEvent, Context<Field, PrevContext>>{
        ReturnControlEvent{.event = Realm::UserEvent::create_user_event()},
        std::move(*this)};
  }

  auto stream_ordered() {
    // a stream ordered context necessarily splits control/execution
    return Context<StreamOrdered, Context<Field, PrevContext>>{StreamOrdered{},
                                                               std::move(*this)}
        .split_control_execution();
  }

  auto after(Realm::UserEvent& ev) && {
    return Context<Realm::Event, Context<Field, PrevContext>>{std::move(ev),
                                                              std::move(*this)};
  }

  auto after(Realm::UserEvent&& ev) && {
    return Context<Realm::Event, Context<Field, PrevContext>>{std::move(ev),
                                                              std::move(*this)};
  }

  auto after(const Realm::UserEvent& ev) && {
    return Context<Realm::Event, Context<Field, PrevContext>>{ev,
                                                              std::move(*this)};
  }

  auto after(Realm::Event&& ev) && {
    return Context<Realm::Event, Context<Field, PrevContext>>{std::move(ev),
                                                              std::move(*this)};
  }

  auto after(Realm::Event& ev) && {
    return Context<Realm::Event, Context<Field, PrevContext>>{std::move(ev),
                                                              std::move(*this)};
  }

  auto after(const Realm::Event& ev) && {
    return Context<Realm::Event, Context<Field, PrevContext>>{ev,
                                                              std::move(*this)};
  }

  template <class T>
  auto after(const ReturnToken<T>& token) && {
    return Context<Realm::Event, Context<Field, PrevContext>>{token.Event(),
                                                              std::move(*this)};
  }

  auto priority(int priority) && {
    return Context<Priority, Context<Field, PrevContext>>{Priority{priority},
                                                          std::move(*this)};
  }

  auto across(DeviceList devices) && {
    return Context<DeviceList, Context<Field, PrevContext>>{std::move(devices),
                                                            std::move(*this)};
  }

  auto region(std::string name) && {
    return Context<std::string, Context<Field, PrevContext>>{std::move(name),
                                                             std::move(*this)};
  }

  auto sync_to(Barrier& barrier) && {
    return Context<std::reference_wrapper<Barrier>,
                   Context<Field, PrevContext>>{barrier, std::move(*this)};
  }
};

template <typename DeferContext, class Fxn, class... Args>
auto _defer(DeferContext context, Fxn f, Args&&... args) {
  using dst_arg_tuple = typename Function<Fxn>::arg_tuple;
  using src_arg_tuple = std::tuple<Args...>;
  using stored_arg_tuple =
      decltype(StoreArgsAndSetPostcondition<dst_arg_tuple, Args...>(
          Realm::Event{}, Realm::Event{}, std::forward<Args>(args)...));
  using raw_ret_type = typename Function<Fxn>::ret_type;
  using tuple_ret_type = typename Tupleify<raw_ret_type>::type;

  Processor p = [&] {
    if constexpr (context.template has_value<Processor>()) {
      return std::move(context.template value<Processor>());
    }
    return Processor::Util();
  }();

  if constexpr (context.template has_value<DeviceList>()) {
    static_assert(!is_tuple<tuple_ret_type>::value,
                  "functions across submesh cannot return a value");
    const auto& devices = context.template value<DeviceList>();
    if (!devices.Contains(p.global_id())) {
      if constexpr (context.template has_value<ReturnControlEvent>()) {
        return ReturnToken{Realm::Event::NO_EVENT,
                           std::tuple<Future<void>>{Realm::Event::NO_EVENT},
                           Realm::UserEvent::NO_USER_EVENT};
      } else {
        return ReturnToken{Realm::Event::NO_EVENT,
                           std::tuple<Future<void>>{Realm::Event::NO_EVENT}};
      }
    }
  }

  auto control_event = [&]() -> std::optional<Realm::UserEvent> {
    if constexpr (context.template has_value<ReturnControlEvent>()) {
      return context.template value<ReturnControlEvent>().event;
    }
    return std::nullopt;
  }();

  const int priority = [&] {
    if constexpr (DeferContext::template has_value<Priority>()) {
      return context.template value<Priority>().value;
    }
    return 0;
  }();

  Realm::UserEvent wait_on_init = Realm::UserEvent::create_user_event();

  auto task_context =
      Context<>::Create()
          .template conditional_move_from<ReturnControlEvent>(context)
          .template conditional_move_from<std::string>(context)
          .template conditional_move_from<StreamOrdered>(context);

  using task_t = Task<Fxn, decltype(task_context), dst_arg_tuple, src_arg_tuple,
                      stored_arg_tuple, tuple_ret_type, 0>;

  void* task_ptr = ::operator new(sizeof(task_t));

  Realm::Event ev = p.RealmProc().spawn(task_t::Id(p.type()), &task_ptr,
                                        sizeof(void*), wait_on_init, priority);

  auto* task = new (task_ptr)
      task_t(ev, std::move(task_context), p, f, std::forward<Args>(args)...);

  Realm::Event precondition = [&] {
    if constexpr (context.template has_value<Realm::Event>()) {
      Realm::Event precondition = context.template value<Realm::Event>();
      if (task->HasPrecondition()) {
        Realm::Event merged = Realm::Event::merge_events(
            std::move(precondition), task->Precondition());
        log_task.debug() << "starting task after " << merged
                         << " from execution precondition " << precondition
                         << " and data precondition " << task->Precondition();
        return merged;
      }
      return precondition;
    }

    if (task->HasPrecondition()) {
      return task->Precondition();
    }

    return Realm::Event::NO_EVENT;
  }();

  if (p.type() == Processor::Type::GPU) {
    // the task cannot be safely deleted until all kernels enqueued have
    // finished
    Processor::Util().RealmProc().spawn(
        (Realm::Processor::TaskFuncID)GlobalTaskId::DELETE_TASK, &task,
        sizeof(void*), ev);
  }

  // all data effects from the result must be resolved before the profiling
  // region is considered done
  auto results = FutureWrappedReturn<tuple_ret_type>{}(*task, std::move(ev));

  Realm::Event all_effects_done_ev = [&] {
    // no deferred arguments and no future return types means the data effects
    // of the task cannot depend on any subtasks so the event ev subsumes all
    // data effects
    if constexpr (!tuple_has_deferred_element<dst_arg_tuple>::value &&
                  std::is_same_v<tuple_ret_type, void>) {
      return ev;
    }
    // we need to be pessimistic and assume that futures/arguments can depend on
    // an arbitrary number of subtasks
    std::set<Realm::Event> all_effects = {ev};
    std::apply(
        [&](auto&... x) { (..., AddRealmEventPrecondition(all_effects, x)); },
        results);
    std::apply(
        [&](auto&... x) { (..., AddRealmEventPostcondition(all_effects, x)); },
        task->Arguments());
    return Realm::Event::merge_events(all_effects);
  }();

  // okay, futures and promises are linked and race condition is removed
  wait_on_init.trigger(precondition);

  if constexpr (DeferContext::template has_value<
                    std::reference_wrapper<Barrier>>()) {
    context.template value<std::reference_wrapper<Barrier>>()
        .get()
        .AddPrecondition(all_effects_done_ev);
  }

  return ReturnToken{std::move(all_effects_done_ev), std::move(results),
                     std::move(control_event)};
};

inline auto region(std::string name) {
  return Context<NoField, NoContext>{{}, {}}.region(std::move(name));
}

template <class... Args>
auto defer(Args&&... args) {
  return Context<NoField, NoContext>{{}, {}}.defer(std::forward<Args>(args)...);
}

template <class... Args>
auto after(Args&&... args) {
  return Context<NoField, NoContext>{{}, {}}.after(std::forward<Args>(args)...);
}

inline auto across(DeviceList devices) {
  return Context<NoField, NoContext>{{}, {}}.across(std::move(devices));
}

inline auto stream_ordered() {
  return Context<NoField, NoContext>{{}, {}}.stream_ordered();
}

inline auto on(Processor p) {
  return Context<NoField, NoContext>{{}, {}}.on(std::move(p));
}

inline auto on(ProcessorGroup group) {
  return Context<NoField, NoContext>{{}, {}}.on(std::move(group));
}

template <class... Conditions>
int stop(Realm::Runtime rt, Conditions&&... cs) {
  std::set<Realm::Event> events;
  (AddRealmEventPrecondition(events, cs), ...);
  Realm::Event last_event = [&] {
    if (events.size() == 1) {
      return *events.begin();
    }
    return Realm::Event::merge_events(events);
  }();
  return Stop(rt, last_event);
}

template <class Fxn>
int program(Fxn f, int argc, char** argv, RealmConfig cfg = {}) {
  Realm::Runtime rt = Init(argc, argv, std::move(cfg));
  auto token = defer(std::move(f));
  return stop(std::move(rt), std::move(token));
}

template <typename Field, typename PrevContext>
template <typename Fxn, typename... Args>
auto Context<Field, PrevContext>::defer(Fxn f, Args&&... args) {
  if constexpr (has_value<ProcessorGroup>()) {
    static_assert(sizeof...(args) == 0,
                  "bulk launch function cannot take arguments");
    // this is a bulk launch
    std::vector<Realm::Event> events;
    for (auto&& proc : this->value<ProcessorGroup>()) {
      auto token = this->drop<ProcessorGroup>().on(proc).defer(f, proc);
      events.push_back(token.Event());
    }
    Realm::Event all_done_ev = Realm::Event::merge_events(events);
    return ReturnToken{std::move(all_done_ev), std::tuple<>{}};
  } else {
    return _defer(*std::move(this), std::move(f), std::forward<Args>(args)...);
  }
}

}  // namespace zuku

#endif
