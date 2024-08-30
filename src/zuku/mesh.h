/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef _POC_SRC_MESH_H_
#define _POC_SRC_MESH_H_

#include <cstdint>
#include <iostream>
#include <numeric>
#include <optional>
#include <string>
#include <vector>

#include "hash.h"

namespace zuku {

class DeviceList {
 public:
  using value_type = int64_t;

  struct Ctor {
    int64_t start{0};
    int64_t num_devices{1};
  };

  class Iterator {
   public:
    using iterator_category = std::forward_iterator_tag;
    using difference_type = int64_t;
    using value_type = int64_t;
    using pointer = int64_t*;
    using reference = int64_t&;

    explicit Iterator(int64_t value) : value_(value) {}

    const int64_t& operator*() const { return value_; }

    bool operator==(const Iterator& other) const {
      return other.value_ == value_;
    }

    bool operator!=(const Iterator& other) const {
      return other.value_ != value_;
    }

    Iterator operator+(size_t offset) { return Iterator(value_ + offset); }

    Iterator& operator++() {
      ++value_;
      return *this;
    }

   private:
    int64_t value_;
  };

  DeviceList() : start_(0), num_devices_(0) {}

  DeviceList(const Ctor& ctor)
      : start_(ctor.start), num_devices_(ctor.num_devices) {}

  int64_t operator[](int64_t idx) const { return start_ + idx; }

  bool Contains(int64_t device_id) const {
    return device_id >= start_ && device_id < stop();
  }

  bool Contains(const DeviceList& other) const {
    return start_ <= other.start_ && stop() >= other.stop();
  }

  static DeviceList Create(int64_t start, int64_t num_devices) {
    return DeviceList(start, num_devices);
  }

  DeviceList slice(int64_t offset, int64_t num_devices) const {
    return DeviceList{{.start = start_ + offset, .num_devices = num_devices}};
  }

  bool operator==(const DeviceList& other) const {
    return start_ == other.start_ && num_devices_ == other.num_devices_;
  }

  std::optional<int64_t> ShardId(int64_t device_id) const {
    if (Contains(device_id)) {
      return device_id - start_;
    }
    return std::nullopt;
  }

  bool operator!=(const DeviceList& other) const {
    return start_ != other.start_ || num_devices_ != other.num_devices_;
  }

  Iterator begin() const { return Iterator(start_); }

  Iterator end() const { return Iterator(start_ + num_devices_); }

  bool empty() const { return num_devices_ == 0; }

  std::vector<int64_t> vector() const {
    std::vector<int64_t> materialized(num_devices_);
    std::iota(materialized.begin(), materialized.end(), start_);
    return materialized;
  }

  int64_t start() const { return start_; }

  int64_t stop() const { return start_ + num_devices_; }

  int64_t size() const { return num_devices_; }

 private:
  DeviceList(int64_t start, int64_t num_devices)
      : start_(start), num_devices_(num_devices) {}

  int64_t start_;
  int64_t num_devices_;
};

inline std::ostream& operator<<(std::ostream& os, const DeviceList& devices) {
  os << "devices[" << devices.start() << "..." << devices.stop() << ")";
  return os;
}

template <typename H>
H AbslHashValue(H h, const DeviceList& dl) {
  return H::combine(std::move(h), dl.start(), dl.stop());
}

}  // namespace zuku

template <>
struct std::hash<zuku::DeviceList> {
  std::size_t operator()(const zuku::DeviceList& devices) const {
    return zuku::hash(devices.start(), devices.size());
  }
};

#endif
