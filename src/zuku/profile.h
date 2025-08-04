/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <string>

#include "processor.h"

namespace zuku {

uint64_t StartProfileRegion(const Processor& p, const std::string& name);

void StopProfileRegion(uint64_t id);

void ProfileRegion(const Processor& p, std::string name, Realm::Event start,
                   Realm::Event stop);

void MarkProfile(const std::string& name);

void MarkProfile(const Processor& p, const std::string& name,
                 const Realm::Event& ev);

void SetupProfiling();

void ShutdownProfiling();

}  // namespace zuku