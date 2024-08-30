/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef _POC_SRC_LINK_PROMISE_H_
#define _POC_SRC_LINK_PROMISE_H_

#include <tuple>

#include "future.h"
#include "task.h"

namespace zuku {

template <class Returned, int Index>
struct LinkFutureAndPromise {
  template <typename Fut, typename Promise>
  void operator()(Fut& f, TaskBase& task, Promise& p) {
    // the function returns a concrete value which should
    // resolve the future
    if (task.HasPrecondition()) {
      f.AddPrecondition(task.Precondition());
    }
  }
};

template <typename T, int Index>
struct LinkFutureAndPromise<Future<T>, Index> {
  template <typename Fut, typename Promise>
  void operator()(Fut& f, TaskBase& task, Promise& p) {
    // the future holds a user event that will need to be resolved
    // before the future can be declared ready
    if (task.HasPrecondition()) {
      f.AddPrecondition(task.Precondition());
    }

    // the function returns a future which is not yet resolved
    // there is no precondtion on the future, it can be passed anywhere
  }
};

template <typename Returned, int Index, class Fut, class Promise>
void link_future_and_promise(Fut& f, TaskBase& t, Promise& p) {
  LinkFutureAndPromise<Returned, Index>{}(f, t, p);
}

template <typename ReturnTuple, int Index, int Remaining>
struct LinkFuturesAndPromises {
  template <typename FutureTuple, typename PromiseTuple>
  void operator()(FutureTuple& futures, TaskBase& task,
                  PromiseTuple& promises) {
    using return_element = std::tuple_element_t<Index, ReturnTuple>;
    link_future_and_promise<return_element, Index>(
        std::get<Index>(futures), task, std::get<Index>(promises));
    LinkFuturesAndPromises<ReturnTuple, Index + 1, Remaining - 1>{}(
        futures, task, promises);
  }
};

template <typename ReturnTuple, int Index>
struct LinkFuturesAndPromises<ReturnTuple, Index, 0> {
  template <typename FutureTuple, typename PromiseTuple>
  void operator()(FutureTuple& futures, TaskBase& task,
                  PromiseTuple& promises) {}
};

}  // namespace zuku

#endif
