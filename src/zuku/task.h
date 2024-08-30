/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef _POC_SRC_TASK_H_
#define _POC_SRC_TASK_H_

#include "realm/event.h"
#include "realm/processor.h"
#include "tiled_array.h"

#include <array>
#include <atomic>
#include <set>
#include <stdexcept>
#include <tuple>

#include "future.h"
#include "init.h"
#include "processor.h"
#include "profile.h"
#include "promise.h"
#include "resolve_dependency.h"
#include "store.h"
#include "task_base.h"
#include "type_traits.h"

namespace zuku {

struct ReturnControlEvent {
  Realm::UserEvent event;
};

struct StreamOrdered {};

extern Realm::Logger log_task;

template <class Tuple, size_t... I>
void PrintTuple(const Tuple& t, std::index_sequence<I...>) {
  std::cout << "(";
  (..., (std::cout << std::get<I>(t) << ","));
  std::cout << ")\n";
}

template <class Tuple>
void PrintTuple(const Tuple& t) {
  PrintTuple(t, std::make_index_sequence<std::tuple_size_v<Tuple>>());
}

void deferred_task_launcher(const void* args, size_t arglen,
                            const void* userdata, size_t userlen,
                            Realm::Processor p);

void deferred_task_delete(const void* args, size_t arglen, const void* userdata,
                          size_t userlen, Realm::Processor p);

void deferred_task_profile_stop(const void* args, size_t arglen,
                                const void* userdata, size_t userlen,
                                Realm::Processor p);

class RealmTaskDescriptor {
 public:
  RealmTaskDescriptor(int64_t next_id, Processor::Type type) {
    Realm::Processor::Kind kind = Processor::ToRealmKind(type);
    Realm::CodeDescriptor descr{deferred_task_launcher};
    Realm::ProfilingRequestSet no_requests;
    Realm::Processor::register_task_by_kind(
        kind, /*global=*/false, /*func_id=*/next_id, descr, no_requests);
  }
};

int64_t nextRealmTaskId();

template <class Fxn, Processor::Type type>
static int64_t GetTaskId() {
  static int64_t next_id = nextRealmTaskId();
  static RealmTaskDescriptor* descr = new RealmTaskDescriptor(next_id, type);
  return next_id;
}

template <class Fxn>
static int64_t GetTaskId(Processor::Type type) {
  switch (type) {
    case Processor::Type::CPU:
      return GetTaskId<Fxn, Processor::Type::CPU>();
    case Processor::Type::GPU:
      return GetTaskId<Fxn, Processor::Type::GPU>();
    case Processor::Type::UTIL:
      return GetTaskId<Fxn, Processor::Type::UTIL>();
    case Processor::Type::TEST:
    case Processor::Type::NUM_TYPES:
      return -1;
  }
  return -1;
}

template <std::size_t Index, std::size_t N>
struct AddTaskDependencies {
  template <class Task>
  void operator()(Task& t) {
    t.template MaybeAddArgPrecondition<Index>();
    AddTaskDependencies<Index + 1, N - 1>{}(t);
  }
};

template <std::size_t Index>
struct AddTaskDependencies<Index, 0> {
  template <class Task>
  void operator()(Task& t) {}
};

template <class T>
auto CanonicalizeArgument(T&& t) {
  return std::forward<T>(t);
}

template <class TargetTuple, std::size_t Offset, std::size_t Index,
          typename CtorArg>
auto make_stored_args_tuple_element(const Realm::Event& stream,
                                    const Realm::Event& effects,
                                    CtorArg&& arg) {
  using target_t = std::tuple_element_t<Index + Offset, TargetTuple>;
  return ToTaskArgument<CtorArg, target_t>{}(stream, effects,
                                             std::forward<CtorArg>(arg));
}

template <class TargetTuple, std::size_t Offset, std::size_t... I,
          typename... CtorArgs>
auto make_stored_args_tuple(const Realm::Event& stream,
                            const Realm::Event& effects,
                            std::index_sequence<I...> seq, CtorArgs&&... args) {
  return std::make_tuple(make_stored_args_tuple_element<TargetTuple, Offset, I>(
      stream, effects, std::forward<CtorArgs>(args))...);
};

template <typename TargetTuple, typename... CtorArgs>
auto StoreArgsAndSetPostcondition(Realm::Event stream, Realm::Event effects,
                                  CtorArgs&&... args) {
  // There can be extra args that going into the lambda that are not explicitly
  // passed in the task arguments. The target tuple is therefore >= ctorargs
  constexpr std::size_t offset =
      std::tuple_size_v<TargetTuple> - sizeof...(CtorArgs);
  return make_stored_args_tuple<TargetTuple, offset>(
      stream, effects, std::make_index_sequence<sizeof...(CtorArgs)>{},
      std::forward<CtorArgs>(args)...);
}

template <class Context>
const Realm::Event& GetStreamOrderedEvent(Context& context,
                                          const Realm::Event& effects) {
  if constexpr (Context::template has_value<StreamOrdered>()) {
    return context.template value<ReturnControlEvent>().event;
  } else {
    return effects;
  }
}

template <class Lambda, class Context, class TargetTuple, class SourceArgsTuple,
          class StoredArgsTuple, class ReturnTuple, int N>
struct Task : public TaskBase {
 public:
  using promise_tuple =
      decltype(PromiseWrappedReturn<ReturnTuple>{}(std::declval<Task&>()));

  template <std::size_t Index>
  using promise_element_t = std::tuple_element_t<Index, promise_tuple>;

  template <std::size_t Index>
  using target_argument_t = std::tuple_element_t<Index, TargetTuple>;

  template <class... CtorArgs>
  Task(Realm::Event completion_ev, Context context, Processor p, Lambda lambda,
       CtorArgs&&... args)
      : TaskBase(std::move(p)),
        lambda_(std::move(lambda)),
        promises_(PromiseWrappedReturn<ReturnTuple>{}(*this)),
        context_(std::move(context)),
        completion_event_(std::move(completion_ev)),
        arguments_(StoreArgsAndSetPostcondition<TargetTuple>(
            GetStreamOrderedEvent<Context>(context_, completion_ev),
            completion_ev, std::forward<CtorArgs>(args)...)) {
    AddTaskDependencies<0, sizeof...(CtorArgs)>{}(*this);
    MergePreconditions();
  }

  ~Task() override = default;

  static int64_t Id(Processor::Type type) { return GetTaskId<Lambda>(type); }

  void Invoke() override {
    uint64_t profile_id = 0;
    if constexpr (Context::template has_value<std::string>()) {
      log_task.debug() << "starting " << context_.template value<std::string>()
                       << " after precondition " << Precondition();
      profile_id =
          StartProfileRegion(proc_, context_.template value<std::string>());
    }

    if constexpr (!is_tuple<ReturnTuple>::value) {
      // the function can take extra arguments that are not included the stored
      // arguments
      constexpr std::size_t offset =
          std::tuple_size_v<TargetTuple> - std::tuple_size_v<StoredArgsTuple>;
      dispatch_void<offset>(
          std::make_index_sequence<std::tuple_size<StoredArgsTuple>{}>{});
    } else {
      dispatch(std::make_index_sequence<std::tuple_size<StoredArgsTuple>{}>{});
    }

    if constexpr (Context::template has_value<ReturnControlEvent>()) {
      if constexpr (Context::template has_value<std::string>()) {
        log_task.debug() << "releasing control on "
                         << context_.template value<std::string>()
                         << " by triggering "
                         << context_.template value<ReturnControlEvent>().event;
      }
      context_.template value<ReturnControlEvent>().event.trigger();
    }

    if constexpr (Context::template has_value<std::string>()) {
      Processor::Util().RealmProc().spawn(
          (Realm::Processor::TaskFuncID)GlobalTaskId::PROFILE_STOP, &profile_id,
          sizeof(uint64_t), completion_event_);
    }
  }

  template <class T>
  void AddDeferredPrecondition(T&& input) {
    if (input.HasPrecondition()) {
      if constexpr (Context::template has_value<StreamOrdered>()) {
        AddPrecondition(input.StreamPrecondition());
      } else {
        AddPrecondition(input.EffectsPrecondition());
      }
    }
  }

  template <std::size_t Index>
  void MaybeAddArgPrecondition() {
    using tuple_element = std::tuple_element_t<Index, StoredArgsTuple>;
    if constexpr (is_deferred<tuple_element>::value) {
      AddDeferredPrecondition(std::get<Index>(arguments_));
    } else if constexpr (is_vector<tuple_element>::value) {
      if constexpr (is_deferred<typename tuple_element::value_type>::value) {
        for (auto&& element : std::get<Index>(arguments_)) {
          AddDeferredPrecondition(element);
        }
      }
    }
  }

  template <std::size_t Index>
  auto& promise() {
    return std::get<Index>(promises_);
  }

  template <std::size_t Index>
  auto& get() & {
    return std::get<Index>(arguments_);
  }

  template <std::size_t Index>
  auto const& get() const& {
    return std::get<Index>(arguments_);
  }

  template <std::size_t Index>
  auto& get() && {
    return std::get<Index>(arguments_);
  }

  template <std::size_t Index>
  auto const& get() const&& {
    return std::get<Index>(arguments_);
  }

  const StoredArgsTuple& Arguments() const { return arguments_; }

 private:
  template <std::size_t... I>
  void dispatch(std::index_sequence<I...>) {
    auto results =
        lambda_(resolve_dependency_tuple<StoredArgsTuple, TargetTuple, 0, I>(
            arguments_)...);
    FulfillPromises<decltype(results), 0, std::tuple_size_v<ReturnTuple>>{}(
        *this, std::move(results));
  }

  template <std::size_t Offset, std::size_t... I>
  void dispatch_void(std::index_sequence<I...>) {
    if constexpr (Context::template has_value<StreamOrdered>()) {
      // awfulness necessary to stack allocate the stream object
      StreamVariant stream_variant;
      auto* stream = [&]() -> Stream* {
        if (proc_.type() == Processor::Type::CPU) {
          return &std::get<CpuStream>(stream_variant);
        }
        stream_variant = GpuStream{};
        return &std::get<GpuStream>(stream_variant);
      }();
      lambda_(stream,
              resolve_dependency_tuple<StoredArgsTuple, TargetTuple, Offset, I>(
                  arguments_)...);
      if (!stream->HasOrderingEvent()) {
        throw std::runtime_error(
            "stream-ordered task did not log completion event on provided "
            "zuku::Stream");
      }
    } else {
      lambda_(resolve_dependency_tuple<StoredArgsTuple, TargetTuple, Offset, I>(
          arguments_)...);
    }
  }

  promise_tuple promises_;
  Lambda lambda_;
  Context context_;
  Realm::Event completion_event_;
  StoredArgsTuple arguments_;
};

}  // namespace zuku

#endif