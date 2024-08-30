/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <realm.h>

#include <iostream>

#include "defer.h"
#include "mesh.h"
#include "processor.h"
#include "reshard.h"
#include "shape.h"
#include "tiled_array.h"
#include "reshard_test.h"
#include "shape_utils.h"

namespace zuku {
namespace {

Realm::Logger log_test("unit-test");

}  // namespace

struct ZeroOut {
  void operator()(ShardedArray& array) const {
    Iterate(array.tile(), [&](int& data, auto... indices) { data = 0; });
  }
};

struct WriteValues {
  void operator()(Processor p, ShardedArray& array) const {
    Iterate(array.tile(), [&](int& data, auto... indices) {
      data = UniqueValue(indices...);
    });
  }
};

struct CheckValues {
  void operator()(Processor p, const ShardedArray& array) const {
    zuku::Iterate(array.tile(), [&](const int& data, auto... indices) {
      const int expected = UniqueValue(indices...);
      if (data != expected) {
        std::cerr << "unexpected value on " << p.global_id()
                  << " in dest data[";
        ((std::cerr << ',' << indices), ...);
        std::cerr << "]: " << data << " != " << expected << " ptr=" << &data
                  << std::endl;
        abort();
      }
    });
  }
};

void run_reshard_test(ShardedShape source_shape, ShardedShape target_shape,
                      DeviceList source_mesh, DeviceList target_mesh,
                      int num_iterations) {
  target_shape.sharding.devices = target_mesh;
  auto waiter = on(ProcessorGroup::Local()).defer([=](Processor p) {
    Store<ShardedArray> source =
        ShardedArray::Create(source_shape, {.processor = p});
    Store<ShardedArray> target =
        ShardedArray::Create(target_shape, {.processor = p});

    const int64_t num_src_elements =
        source->shape().sharding.NumLocalElements();
    const int64_t num_dst_elements =
        target->shape().sharding.NumLocalElements();

    for (int64_t iter = 0; iter < num_iterations; ++iter) {
      across(source_mesh).if_on(p).defer(WriteValues{}, p, source);

      Reshard(p, source, target);

      across(target_mesh).if_on(p).defer(CheckValues{}, p, target);

      across(target_mesh).if_on(p).defer(ZeroOut{}, target);
      across(source_mesh).if_on(p).defer(ZeroOut{}, source);
    }
    source.Wait();
    target.Wait();
  });
  waiter.Wait();
}

}  // namespace zuku