/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef _POC_SRC_BARRIER_H_
#define _POC_SRC_BARRIER_H_

#include "realm.h"

#include <optional>
#include <stdexcept>
#include <vector>

namespace zuku {

class Barrier {
 public:
  void AddPrecondition(Realm::Event ev) { events_.push_back(std::move(ev)); }
  void Wait() { Realm::Event::merge_events(events_).wait(); }

 private:
  std::vector<Realm::Event> events_;
};

}  // namespace zuku

#endif
