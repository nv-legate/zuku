/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "processor.h"

#include "realm/machine.h"
#include "realm/memory.h"
#include "realm/processor.h"

#include "mesh.h"

namespace zuku {

Realm::Logger log_proc("zuku-proc");

namespace {

Processor::Type default_proc_type{Processor::Type::CPU};
std::map<Processor::Type, std::vector<Processor>> local_procs;
std::map<Processor::Type, std::vector<Realm::Processor>> all_procs;

struct RemoteProcessors {
  int64_t global_device_id_start;
  int64_t global_device_id_stop;
  std::vector<Realm::Processor> processors;
};
std::map<Processor::Type, std::vector<RemoteProcessors>>
    all_procs_by_address_space;

}  // namespace

Realm::Processor::Kind Processor::ToRealmKind(Processor::Type type) {
  switch (type) {
    case Type::CPU:
      return Realm::Processor::LOC_PROC;
    case Type::GPU:
      return Realm::Processor::TOC_PROC;
    case Type::UTIL:
      return Realm::Processor::UTIL_PROC;
    case Type::NUM_TYPES:
      return Realm::Processor::NO_KIND;
  }
  return Realm::Processor::NO_KIND;
}

Realm::Memory::Kind Processor::DefaultMemoryKind() const {
  switch (type_) {
    case Processor::Type::GPU:
      return Realm::Memory::Kind::GPU_FB_MEM;
    default:
      return Realm::Memory::Kind::SYSTEM_MEM;
  }
}

Processor::Type Processor::DefaultType() { return default_proc_type; }

Processor Processor::Default() { return Processor::Create({.local = 0}); }

Processor Processor::Util() {
  return Processor::Create({.local = 0}, Processor::Type::UTIL);
}

int64_t Processor::NumLocalInDeviceList(const DeviceList& devices,
                                        Processor::Type type) {
  int64_t num_local = 0;
  for (auto& proc : local_procs[type]) {
    if (devices.Contains(proc.global_id())) {
      ++num_local;
    }
  }
  return num_local;
}

void Processor::Init() {
  for (auto type : {Type::UTIL, Type::CPU, Type::GPU}) {
    Realm::Processor::Kind kind = Processor::ToRealmKind(type);

    auto machine = Realm::Machine::get_machine();
    auto& procs_by_address_space = all_procs_by_address_space[type];
    procs_by_address_space.resize(machine.get_address_space_count());
    int64_t total_procs = 0;
    Realm::Machine::ProcessorQuery procs_query(machine);
    procs_query.only_kind(kind);
    for (Realm::Machine::ProcessorQuery::iterator it = procs_query.begin();
         it != procs_query.end(); it++) {
      procs_by_address_space[it->address_space()].processors.push_back(*it);
      log_proc.debug() << "have processor " << *it << " on address space "
                       << it->address_space();
      ++total_procs;
    }

    auto& all_procs_for_kind = all_procs[type];
    all_procs_for_kind.reserve(total_procs);
    // now sort within the address spaces to create a consistent numbering of
    // the procs
    int64_t global_offset = 0;
    for (auto& addr_space_procs : procs_by_address_space) {
      std::sort(addr_space_procs.processors.begin(),
                addr_space_procs.processors.end());
      all_procs_for_kind.insert(all_procs_for_kind.end(),
                                addr_space_procs.processors.begin(),
                                addr_space_procs.processors.end());
      addr_space_procs.global_device_id_start = global_offset;
      global_offset += addr_space_procs.processors.size();
      addr_space_procs.global_device_id_stop = global_offset;
    }

    auto& local_procs_for_kind = local_procs[type];
    std::set<Realm::Processor> realm_procs;
    Realm::Machine::get_machine().get_local_processors_by_kind(realm_procs,
                                                               kind);
    local_procs_for_kind.reserve(realm_procs.size());
    if (!realm_procs.empty()) {
      int64_t local_id = 0;
      int64_t global_id =
          procs_by_address_space[realm_procs.begin()->address_space()]
              .global_device_id_start;
      for (auto&& realm_proc : realm_procs) {
        log_proc.debug() << "next processor is " << realm_proc
                         << " on address space " << realm_proc.address_space()
                         << " assigned to local_id=" << local_id
                         << ", global_id=" << global_id
                         << " of type=" << (int)type;
        local_procs_for_kind.push_back(
            Processor{local_id++, global_id++, type, realm_proc});
      }
    }
    if (!local_procs_for_kind.empty()) {
      default_proc_type = type;
    }
  }
}

Realm::Processor Processor::GetAttachedHost(
    int64_t global_device_id, std::optional<Processor::Type> type) {
  Realm::Processor device_proc = RemoteProc(global_device_id, std::move(type));
  log_proc.debug() << "getting attached host for global_device "
                   << global_device_id << ", found remote " << device_proc;
  if (device_proc.kind() == Realm::Processor::Kind::LOC_PROC) {
    return device_proc;
  }

  auto& hosts_on_remote =
      all_procs_by_address_space[Processor::Type::CPU]
                                [device_proc.address_space()];
  // if cpus and other type are aligned globally, return the CPU that matches
  if (global_device_id >= hosts_on_remote.global_device_id_start &&
      global_device_id < hosts_on_remote.global_device_id_stop) {
    const int64_t local_device_id =
        global_device_id - hosts_on_remote.global_device_id_start;
    return hosts_on_remote.processors[local_device_id];
  }

  return hosts_on_remote.processors[0];
}

Realm::Processor Processor::RemoteProc(int64_t global_device_id,
                                       std::optional<Processor::Type> type) {
  return all_procs[type.value_or(DefaultType())][global_device_id];
}

const std::vector<Processor>& GetProcsForType(Processor::Type type) {
  return local_procs[type];
}

Processor GetProc(DeviceId id, Processor::Type type) {
  const auto& procs = GetProcsForType(type);
  const int64_t local_id = [&] {
    if (id.local.has_value()) {
      return *id.local;
    }
    // TODO: translate global to locate
    return *id.global;
  }();

  if (local_id >= procs.size()) {
    throw std::runtime_error("local_id is greater than number of local procs");
  }

  return procs[local_id];
}

int64_t Processor::NumForType(Type type) {
  return GetProcsForType(type).size();
}

const std::vector<Processor>& Processor::DefaultProcs() {
  return GetProcsForType(default_proc_type);
}

Processor Processor::Create(DeviceId id, std::optional<Type> type) {
  if (type.has_value() && *type == Processor::Type::TEST) {
    return Processor(*id.local, *id.global, *type, Realm::Processor::NO_PROC);
  }
  Realm::Processor::Kind kind = ToRealmKind(type.value_or(DefaultType()));
  return GetProc(id, type.value_or(DefaultType()));
}

ProcessorGroup ProcessorGroup::Local(std::optional<Processor::Type> type) {
  return ProcessorGroup(local_procs[type.value_or(Processor::DefaultType())]);
}

ProcessorGroup ProcessorGroup::Create(DeviceIds ids,
                                      std::optional<Processor::Type> type) {
  Processor::Type proc_type = type.value_or(Processor::DefaultType());
  std::vector<Processor> procs;
  if (ids.local.has_value()) {
    procs.reserve(ids.local->size());
    for (int64_t id : *ids.local) {
      procs.push_back(Processor::Create({.local = id}, type));
    }
  } else if (ids.global.has_value()) {
    procs.reserve(ids.global->size());
    for (int64_t id : *ids.global) {
      procs.push_back(Processor::Create({.global = id}, type));
    }
  } else {
    throw std::runtime_error("must specifiy local or global IDs");
  }
  return ProcessorGroup(std::move(procs));
}

}  // namespace zuku