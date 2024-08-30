/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef _POC_SRC_STREAM_H_
#define _POC_SRC_STREAM_H_

#include <variant>

namespace zuku {

// opaque type used to avoid have cuda types in the header file
struct EventStream;

class Stream {
 public:
  void Record(EventStream* event) {
    DoRecord(event);
    synced_ = true;
  }

  bool HasOrderingEvent() const { return synced_; }

 protected:
  Stream() : synced_{false} {}
  virtual void DoRecord(EventStream* event) = 0;

 private:
  bool synced_{};
};

class CpuStream : public Stream {
 protected:
  void DoRecord(EventStream* event) override {}
};

class GpuStream : public Stream {
 protected:
  void DoRecord(EventStream* stream) override;
};

using StreamVariant = std::variant<CpuStream, GpuStream>;

}  // namespace zuku

#endif