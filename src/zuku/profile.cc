/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "profile.h"

#include <chrono>
#include <thread>
#include <vector>

#include "defer.h"
#include "init.h"
#include "processor.h"

#ifdef zuku_HAS_NVTX
#include <nvtx3/nvToolsExt.h>
#endif

namespace zuku {
namespace {

bool sProfilingActive{false};
Realm::Logger log_profile("zuku-profile");

#ifdef zuku_HAS_NVTX
nvtxDomainHandle_t zukuDomainHandle;
std::array<std::vector<nvtxEventAttributes_t>,
           std::size_t(Processor::Type::NUM_TYPES)>
    nvtx_attrs;
#endif

}  // namespace

void MarkProfile(const Processor& p, const std::string& name,
                 const Realm::Event& ev) {
  if (!sProfilingActive) {
    return;
  }
#ifdef zuku_HAS_NVTX
  on(Processor::Util()).after(ev).defer([=] {
    std::size_t color_hash = std::hash<std::string>{}(name) % 0xFFFFFF;
    nvtxEventAttributes_t& attr =
        nvtx_attrs[std::size_t(p.type())][p.local_id()];
    attr.message.ascii = name.c_str();
    attr.color = 0xFF000000 | color_hash;
    nvtxDomainMarkEx(zukuDomainHandle, &attr);
  });
#endif
}

void ProfileRegion(const Processor& p, std::string name, Realm::Event start,
                   Realm::Event stop) {
  if (!sProfilingActive) {
    return;
  }
#ifdef zuku_HAS_NVTX
  IncPendingOp();
  auto [profile_id] =
      after(start)
          .on(Processor::Util())
          .defer([=](Processor p,
                     std::string name) { return StartProfileRegion(p, name); },
                 p, std::move(name));

  log_profile.debug() << "will finish profiling after " << stop;

  auto done = after(stop)
                  .on(Processor::Util())
                  .defer(
                      [&](int64_t id) {
                        StopProfileRegion(id);
                        log_profile.debug() << "done profiling " << id;
                        DecPendingOp();
                      },
                      std::move(profile_id));
#endif
}

uint64_t StartProfileRegion(const Processor& p, const std::string& name) {
  if (!sProfilingActive) {
    return -1;
  }
#ifdef zuku_HAS_NVTX
  static_assert(std::is_same_v<uint64_t, nvtxRangeId_t>,
                "nvtxRangeId_t matches");
  std::size_t color_hash = std::hash<std::string>{}(name) % 0xFFFFFF;
  nvtxEventAttributes_t& attr = nvtx_attrs[std::size_t(p.type())][p.local_id()];
  attr.message.ascii = name.c_str();
  attr.color = 0xFF000000 | color_hash;
  return nvtxDomainRangeStartEx(zukuDomainHandle, &attr);
#else
  return 0;
#endif
}

void StopProfileRegion(uint64_t id) {
  if (!sProfilingActive) {
    return;
  }
#ifdef zuku_HAS_NVTX
  nvtxDomainRangeEnd(zukuDomainHandle, id);
#endif
}

void ShutdownProfiling() {}

void SetupProfiling() {
  sProfilingActive = true;
#ifdef zuku_HAS_NVTX
  zukuDomainHandle = nvtxDomainCreateA("zuku");
  for (auto type :
       {Processor::Type::UTIL, Processor::Type::CPU, Processor::Type::GPU}) {
    nvtx_attrs[std::size_t(type)].resize(Processor::NumForType(type));
    for (auto&& attr : nvtx_attrs[std::size_t(type)]) {
      attr.version = NVTX_VERSION;
      attr.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
      attr.colorType = NVTX_COLOR_ARGB;
      attr.color = 0xFFFF0000;
      attr.messageType = NVTX_MESSAGE_TYPE_ASCII;
    }
  }
#endif
}

void TearDownProfiling() {
#ifdef zuku_HAS_NVTX
  nvtxDomainDestroy(zukuDomainHandle);
#endif
}

}  // namespace zuku
