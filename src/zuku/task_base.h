/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef _POC_SRC_TASK_BASE_H_
#define _POC_SRC_TASK_BASE_H_

#include "realm.h"

#include <optional>
#include <set>

#include "processor.h"
#include "type_traits.h"

namespace zuku {

struct TaskBase {
 public:
  virtual void Invoke() = 0;

  virtual ~TaskBase() = default;

  bool HasPrecondition() const { return merged_precondition_.has_value(); }

  const Realm::Event& Precondition() const { return *merged_precondition_; }

  const Processor& processor() const { return proc_; }

 protected:
  TaskBase(Processor p) : proc_(p) {}

  void AddPrecondition(const Realm::Event& event) {
    individual_preconditions_.insert(event);
  }

  void AddPrecondition(Realm::Event&& event) {
    individual_preconditions_.insert(std::move(event));
  }

  void MergePreconditions() {
    if (merged_precondition_.has_value()) {
      throw std::runtime_error("already merged preconditions");
    }

    if (individual_preconditions_.empty()) {
      return;  // leave empty
    } else if (individual_preconditions_.size() == 1) {
      merged_precondition_ = *individual_preconditions_.begin();
    } else {
      merged_precondition_ =
          Realm::Event::merge_events(individual_preconditions_);
    }
    individual_preconditions_.clear();
  }

  std::set<Realm::Event> individual_preconditions_;
  std::optional<Realm::Event> merged_precondition_;
  Processor proc_;
};

template <typename CtorArg, typename TargetArg>
struct ToTaskArgument {
  auto operator()(const Realm::Event& stream, const Realm::Event& effects,
                  CtorArg&& arg) const {
    return std::forward<CtorArg>(arg);
  };
};

}  // namespace zuku

#endif