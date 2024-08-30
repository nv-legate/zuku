/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "init.h"

#include "realm/logging.h"
#include "realm/machine.h"
#include "realm/processor.h"
#include <unistd.h>

#include <array>
#include <stdexcept>
#include <thread>
#include <variant>

#include "processor.h"
#include "profile.h"
#include "reshard_manager.h"
#include "task.h"
#include "type_traits.h"

namespace zuku {
namespace {

Realm::Logger log_init("zuku-init");

constexpr int64_t kMB = 1000 * 1000;

std::atomic<int64_t> sPendingEvents{0};

void check_allocation_task(const void* args, size_t arglen,
                           const void* user_data, size_t userlen,
                           Realm::Processor p) {
  Realm::ProfilingResponse response(args, arglen);
  auto* alloc =
      *static_cast<const AllocationResult* const*>(response.user_data());
  Realm::ProfilingMeasurements::InstanceAllocResult result;
  const bool measured = response.get_measurement(result);
  if (!result.success) {
    alloc->Fail();
  }
  alloc->Done();
}

void WaitForPendingOps() {
  int64_t pending;
  while ((pending = sPendingEvents.load()) > 0) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
}

}  // namespace

void IncPendingOp() {
  const int64_t pending = sPendingEvents.fetch_add(int64_t(1));
}

void DecPendingOp() {
  const int64_t pending = sPendingEvents.fetch_add(int64_t(-1));
}

Realm::Runtime Init(int argc, char** argv, RealmConfig cfg) {
  Realm::Runtime rt;

  if (!rt.create_configs(argc, argv)) {
    throw std::runtime_error("unable to create Realm configs");
  }

  auto network = std::visit(
      overloaded{[](const auto& network) -> std::optional<std::string> {
        using var_t = std::decay_t<decltype(network)>;
        if constexpr (std::is_same_v<var_t, bool>) {
          if (network == true) {
            return std::nullopt;
          }
          return "none";
        } else if constexpr (std::is_same_v<var_t, std::monostate>) {
          return std::nullopt;
        } else {
          if (network == "default") {
            return std::nullopt;
          }
          return network;
        }
      }},
      cfg.network);

  if (network.has_value()) {
    int argc = 3;
    std::array<const char*, 3> argv_array = {"", "-ll:networks",
                                             network->c_str()};
    char** argv = (char**)argv_array.data();
    if (!rt.network_init(&argc, &argv)) {
      throw std::runtime_error("unable to init Realm network");
    }
  } else {
    if (!rt.network_init(&argc, &argv)) {
      throw std::runtime_error("unable to init Realm network");
    }
  }

  auto* core = rt.get_module_config("core");
  if (!core) {
    throw std::runtime_error("no core config found for Realm");
  }
  if (!core->set_property("util", 2)) {
    throw std::runtime_error("failed to set no. util procs");
  }
  if (cfg.cpus.has_value()) {
    if (!core->set_property("cpu", *cfg.cpus)) {
      throw std::runtime_error("unable to set no. cpus on Realm config");
    }
  }

  if (cfg.sysmem.has_value() &&
      !core->set_property("sysmem", *cfg.sysmem * kMB)) {
    throw std::runtime_error("unable to set sysmem on Realm config");
  }

  auto* cuda = rt.get_module_config("cuda");
  if (cuda) {
    if (cfg.zcmem.has_value() &&
        !cuda->set_property("zcmem", *cfg.zcmem * kMB)) {
      throw std::runtime_error("unable to set zcmem on Realm config");
    }
    if (cfg.gpus.has_value() && !cuda->set_property("gpu", *cfg.gpus)) {
      throw std::runtime_error("unable to set no. gpus on Realm config");
    }
    if (cfg.fbmem.has_value() &&
        !cuda->set_property("fbmem", *cfg.fbmem * kMB)) {
      throw std::runtime_error("unable to set fbmem on Realm config");
    }
  } else if (cfg.gpus.value_or(0) > 0) {
    std::cerr << *cfg.gpus << " GPU requested, but CUDA module not found"
              << std::endl;
    abort();
  }

  std::vector<std::string> cmdline{argv + 1, argv + argc};
  cmdline.insert(cmdline.end(), cfg.argv.begin(), cfg.argv.end());
  if (cfg.kthreads) {
    cmdline.push_back("-ll:force_kthreads");
  }
  if (cfg.cmdline.has_value()) {
    cmdline.insert(cmdline.end(), cfg.cmdline->begin(), cfg.cmdline->end());
  }

  rt.parse_command_line(cmdline, true);
  rt.finish_configure();
  rt.start();

  Processor::Init();
  if (cfg.profile) {
    SetupProfiling();
  }

  Realm::Processor p =
      Realm::Machine::ProcessorQuery(Realm::Machine::get_machine())
          .only_kind(Realm::Processor::LOC_PROC)
          .first();

  Realm::CodeDescriptor pt2pt_descr{ReshardManager::PointToPointNameHandler};
  Realm::ProfilingRequestSet no_requests;
  const Realm::Event e1 = Realm::Processor::register_task_by_kind(
      Realm::Processor::Kind::LOC_PROC, /*global=*/false,
      (Realm::Processor::TaskFuncID)
          GlobalTaskId::RESHARD_POINT_TO_POINT_NAME_EXCHANGE,
      pt2pt_descr, no_requests);

  Realm::CodeDescriptor delete_task_descr{deferred_task_delete};
  const Realm::Event e2 = Realm::Processor::register_task_by_kind(
      Realm::Processor::Kind::UTIL_PROC, /*global=*/false,
      (Realm::Processor::TaskFuncID)GlobalTaskId::DELETE_TASK,
      delete_task_descr, no_requests);

  Realm::CodeDescriptor stop_profile_task_descr{deferred_task_profile_stop};
  const Realm::Event e3 = Realm::Processor::register_task_by_kind(
      Realm::Processor::Kind::UTIL_PROC, /*global=*/false,
      (Realm::Processor::TaskFuncID)GlobalTaskId::PROFILE_STOP,
      stop_profile_task_descr, no_requests);

  Realm::CodeDescriptor release_shared_ptr{deferred_release_shared_pointer};
  const Realm::Event e4 = Realm::Processor::register_task_by_kind(
      Realm::Processor::Kind::UTIL_PROC, /*global=*/false,
      (Realm::Processor::TaskFuncID)GlobalTaskId::RELEASE_SHARED_POINTER,
      release_shared_ptr, no_requests);

  Realm::CodeDescriptor check_allocation_descr{check_allocation_task};
  const Realm::Event e5 = Realm::Processor::register_task_by_kind(
      Realm::Processor::UTIL_PROC, /*global=*/false,
      (Realm::Processor::TaskFuncID)GlobalTaskId::CHECK_MEMORY_ALLOCATION,
      check_allocation_descr, no_requests);

  rt.collective_spawn(p, Realm::Processor::TASK_ID_PROCESSOR_NOP, nullptr, 0,
                      Realm::Event::merge_events(e1, e2, e3, e4))
      .wait();

  Realm::Machine machine = Realm::Machine::get_machine();
  Realm::AddressSpace rank =
      Realm::Processor::get_executing_processor().address_space();
  Realm::Machine::MemoryQuery mem_query(machine);
  mem_query.local_address_space();
  for (const auto& mem : mem_query) {
    log_init.debug() << "Memory " << mem << ": " << mem.kind()
                     << ", capacity=" << mem.capacity();
  }

  return rt;
}

int Stop(Realm::Runtime& rt, const Realm::Event& last_local_event) {
  Realm::Processor p =
      Realm::Machine::ProcessorQuery(Realm::Machine::get_machine())
          .only_kind(Realm::Processor::LOC_PROC)
          .first();

  WaitForPendingOps();

  Realm::Event agree = rt.collective_spawn(
      p, Realm::Processor::TASK_ID_PROCESSOR_NOP, nullptr, 0, last_local_event);

  agree.wait();

  // give Realm another second to clean up other pending operations
  sleep(1);

  rt.shutdown(agree);
  return rt.wait_for_shutdown();
}

}  // namespace zuku
