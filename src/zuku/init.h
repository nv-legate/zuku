/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef _POC_SRC_INIT_H_
#define _POC_SRC_INIT_H_

#include "realm/event.h"
#include "realm.h"
#include "realm/processor.h"
#include "realm/runtime.h"

#include <optional>
#include <variant>

namespace zuku {

enum class GlobalTaskId {
  RESHARD_POINT_TO_POINT_NAME_EXCHANGE =
      Realm::Processor::TASK_ID_FIRST_AVAILABLE + 4,
  RESHARD_SCATTER_GATHER_NAME_EXCHANGE,
  DELETE_TASK,
  PROFILE_STOP,
  RELEASE_SHARED_POINTER,
  CHECK_MEMORY_ALLOCATION,
  FIRST_AVAILABLE,
};

struct RealmConfig {
  std::optional<int> cpus{std::nullopt};
  std::optional<int> gpus{std::nullopt};
  std::optional<int64_t> sysmem{std::nullopt};
  std::optional<int64_t> fbmem{std::nullopt};
  std::optional<int64_t> zcmem{std::nullopt};
  std::variant<std::monostate, std::string, bool> network{true};
  std::optional<std::vector<std::string>> cmdline;
  bool kthreads{false};
  bool profile{true};
  std::vector<std::string> argv{};
};

class AllocationResult {
 private:
  Realm::UserEvent signal{Realm::UserEvent::NO_USER_EVENT};
  mutable std::atomic<bool> success{true};

 public:
  AllocationResult() : signal(Realm::UserEvent::create_user_event()) {}

  void Wait() { signal.wait(); }

  void Done() const { signal.trigger(); }

  void Fail() const { success.store(false); }

  bool Succeeded() const { return success.load(); }
};

void IncPendingOp();
void DecPendingOp();

Realm::Runtime Init(int argc, char** argv, RealmConfig cfg);

int Stop(Realm::Runtime& rt, const Realm::Event& last_local_event);

}  // namespace zuku

#endif
