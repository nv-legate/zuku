/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "task.h"

#include "future.h"
#include "init.h"

namespace zuku {

Realm::Logger log_task("zuku-task");

int64_t nextRealmTaskId() {
  static std::atomic<int64_t> id{int64_t(GlobalTaskId::FIRST_AVAILABLE)};
  return id.fetch_add(int64_t(1));
}

void deferred_task_launcher(const void* args, size_t arglen,
                            const void* userdata, size_t userlen,
                            Realm::Processor p) {
  TaskBase* task = *const_cast<TaskBase**>(static_cast<TaskBase* const*>(args));
  task->Invoke();
  if (task->processor().type() != Processor::Type::GPU) {
    // if on a "synchronous" processor, go ahead and delete the task
    // all work has been completed
    // if on an "asynchronous" processor, we cannot guarantee the work has
    // been completed and that means task deletion has to be scheduled
    // after the task is done
    delete task;
  }
}

void deferred_task_delete(const void* args, size_t arglen, const void* userdata,
                          size_t userlen, Realm::Processor p) {
  TaskBase* task = *const_cast<TaskBase**>(static_cast<TaskBase* const*>(args));
  delete task;
}

void deferred_task_profile_stop(const void* args, size_t arglen,
                                const void* userdata, size_t userlen,
                                Realm::Processor p) {
  auto* id_ptr = static_cast<const uint64_t*>(args);
  StopProfileRegion(*id_ptr);
}

void deferred_release_shared_pointer(const void* args, size_t arglen,
                                     const void* userdata, size_t userlen,
                                     Realm::Processor p) {
  auto* holder = *static_cast<ReleaseHolderBase* const*>(args);
  log_task.debug() << "releasing " << holder << " after event "
                   << holder->event();
  delete holder;
  DecPendingOp();
}

}  // namespace zuku