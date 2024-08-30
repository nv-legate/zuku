/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef _POC_SRC_INSTANCE_CACHE_H_
#define _POC_SRC_INSTANCE_CACHE_H_

#include "realm/memory.h"

#include <queue>
#include <unordered_map>

#include "processor.h"
#include "store.h"

namespace zuku {

template <class Key, class Value>
class InstanceCache {
 public:
  using factory_t =
      std::function<Store<Value>(Key, Realm::Memory::Kind, Processor)>;

  InstanceCache(Realm::Memory::Kind memory, Processor p, factory_t factory)
      : memory_(memory), processor_(p), factory_(std::move(factory)) {}

  Store<Value> Get(const Key& key,
                   std::optional<int64_t> min_cache_size = std::nullopt) {
    auto& cache = values_[key];
    if (cache.size() < min_cache_size.value_or(1)) {
      return factory_(key, memory_, processor_);
    }

    Store<Value> value = std::move(cache.front());
    cache.pop();
    return value;
  }

  auto Make(const Key& key) { return factory_(key, memory_, processor_); }

  void Clear() { values_.clear(); }

  void Free(const Key& key, Store<Value> value) {
    values_[key].push(std::move(value));
  }

  Processor processor() const { return processor_; }

  Realm::Memory::Kind memory() const { return memory_; }

 private:
  std::unordered_map<Key, std::queue<Store<Value>>> values_;
  Realm::Memory::Kind memory_;
  Processor processor_;
  factory_t factory_;
};

}  // namespace zuku

#endif  // _POC_SRC_INSTANCE_CACHE_H_