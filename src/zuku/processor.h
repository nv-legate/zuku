/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef _POC_SRC_PROCESSOR_H_
#define _POC_SRC_PROCESSOR_H_

#include "realm.h"
#include "realm/memory.h"

#include <cstdint>
#include <optional>
#include <vector>

#include "mesh.h"
#include "stream.h"

namespace zuku {

extern Realm::Logger log_proc;

struct DeviceId {
  std::optional<int64_t> local;
  std::optional<int64_t> global;
};

struct DeviceIds {
  std::optional<DeviceList> local;
  std::optional<DeviceList> global;
};

class Processor {
 public:
  enum class Type { UTIL, CPU, GPU, TEST, NUM_TYPES };

  static Realm::Processor::Kind ToRealmKind(Processor::Type type);

  int64_t local_id() const { return local_id_; }

  int64_t global_id() const { return global_id_; }

  DeviceId id() const {
    return DeviceId{.local = local_id_, .global = global_id_};
  }

  Realm::Memory::Kind DefaultMemoryKind() const;

  const Realm::Processor& RealmProc() const& { return proc_; }

  Realm::Processor&& RealmProc() && { return std::move(proc_); }

  Type type() const { return type_; }

  static Type DefaultType();

  static int64_t NumForType(Type type);

  static void Init();

  static Processor Default();

  static Processor Util();

  static int64_t NumLocalInDeviceList(const DeviceList& devices,
                                      Processor::Type type);

  static Realm::Processor RemoteProc(
      int64_t global_device_id,
      std::optional<Processor::Type> type = std::nullopt);

  static Realm::Processor GetAttachedHost(
      int64_t global_device_id,
      std::optional<Processor::Type> type = std::nullopt);

  static const std::vector<Processor>& DefaultProcs();

  static Processor Create(DeviceId id, std::optional<Type> = std::nullopt);

 private:
  friend const std::vector<Processor>& GetProcsForType(Type);

  Processor(int64_t local_id, int64_t global_id, Type type, Realm::Processor p)
      : type_(type), local_id_(local_id), global_id_(global_id), proc_(p) {}

  Type type_;
  Realm::Processor proc_;
  int64_t local_id_;
  int64_t global_id_;
};

class ProcessorGroup {
 public:
  static ProcessorGroup Create(DeviceIds ids,
                               std::optional<Processor::Type> = std::nullopt);

  auto begin() const { return procs_.begin(); }

  auto begin() { return procs_.begin(); }

  auto end() const { return procs_.end(); }

  auto end() { return procs_.end(); }

  static ProcessorGroup Local(
      std::optional<Processor::Type> type = std::nullopt);

 private:
  explicit ProcessorGroup(std::vector<Processor> procs)
      : procs_(std::move(procs)) {}

  std::vector<Processor> procs_;
};

inline bool operator==(const Processor& lhs, const Processor& rhs) {
  return lhs.type() == rhs.type() && lhs.global_id() == rhs.global_id();
}

inline std::ostream& operator<<(std::ostream& os, const Processor& p) {
  os << "Processor(type=" << int(p.type()) << ",global_id=" << p.global_id()
     << ")";
  return os;
}

}  // namespace zuku

template <>
struct std::hash<zuku::Processor> {
  std::size_t operator()(const zuku::Processor& p) const {
    return zuku::hash((int)p.type(), p.global_id());
  }
};

#endif
