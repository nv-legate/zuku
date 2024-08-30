/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <realm.h>
#include <realm/logging.h>

#include <iostream>

#include "defer.h"
#include "mesh.h"
#include "processor.h"
#include "reshard.h"
#include "shape.h"
#include "tiled_array.h"

using zuku::DeviceList;
using zuku::Processor;
using zuku::ProcessorGroup;
using zuku::ShardedArray;
using zuku::ShardedShape;
using zuku::ShardingDim;
using zuku::Store;

Realm::Logger log_test("test");

ShardedShape GetShape(DeviceList devices, int64_t global_size) {
  ShardedShape shape{
      .type = zuku::SupportedType::F32,
      .sharding =
          {
              .dims = {ShardingDim{
                  .size = global_size,
                  .sharding = devices.size(),
              }},
              .devices = std::move(devices),
          },
  };
  return shape;
}

constexpr int64_t kTotalDevices = 4;
constexpr int64_t kTaskNumDevices = kTotalDevices / 2;
constexpr int64_t kLocalArraySize = 4;
constexpr int64_t kGlobalArraySize = kLocalArraySize * kTaskNumDevices;
constexpr int64_t kNumIterations = 5;

struct ZeroOut {
  void operator()(ShardedArray& array) const {
    int* data = array.tile().ptr<int>();
    for (int64_t i = 0; i < kLocalArraySize; ++i) {
      data[i] = 0;
    }
  }
};

struct CheckValues {
  void operator()(Processor p, const ShardedArray& array, int offset) const {
    const int* data = array.tile().ptr<int>();
    for (int64_t i = 0; i < kLocalArraySize; ++i) {
      const int multiplier = (p.global_id() - offset) % kTaskNumDevices + 1;
      for (int64_t i = 0; i < kLocalArraySize; ++i) {
        const int expected = multiplier * i;
        if (data[i] != expected) {
          std::cerr << "unexpected value in dest data on " << p.global_id()
                    << ": " << data[i] << " != " << expected << std::endl;
          abort();
        }
      }
    }
    log_test.debug() << "Values are correct on " << p.global_id();
  }
};

auto test(DeviceList first_mesh, DeviceList second_mesh) {
  auto waiter = on(ProcessorGroup::Local()).defer([=](Processor p) {
    Store<ShardedArray> source = ShardedArray::Create(
        GetShape(first_mesh, kGlobalArraySize), {.processor = p});
    Store<ShardedArray> dest = ShardedArray::Create(
        GetShape(second_mesh, kGlobalArraySize), {.processor = p});

    const int offset = second_mesh.start();
    for (int64_t iter = 0; iter < kNumIterations; ++iter) {
      across(first_mesh)
          .if_on(p)
          .defer(
              [=](ShardedArray& array) {
                int* data = array.tile().ptr<int>();
                for (int64_t i = 0; i < kLocalArraySize; ++i) {
                  data[i] = i * (p.global_id() + 1);
                }
              },
              source);

      Reshard(p, source, dest);

      across(second_mesh).if_on(p).defer(CheckValues{}, p, dest, offset);

      // zero out the source data
      across(first_mesh).if_on(p).defer(ZeroOut{}, source);

      // write the values back from dest to source
      Reshard(p, dest, source);

      // check the valus on both
      across(first_mesh).if_on(p).defer(CheckValues{}, p, source, /*offset=*/0);

      across(second_mesh).if_on(p).defer(CheckValues{}, p, dest, offset);

      // zero out for the next iteration to clear the data
      across(first_mesh).if_on(p).defer(ZeroOut{}, source);

      across(second_mesh).if_on(p).defer(ZeroOut{}, dest);
    }
    source.Wait();
    dest.Wait();
  });
  waiter.Wait();
}

auto run() {
  {
    DeviceList first_mesh{{.start = 0, .num_devices = kTaskNumDevices}};
    DeviceList second_mesh{
        {.start = kTaskNumDevices, .num_devices = kTaskNumDevices}};
    test(first_mesh, second_mesh);
  }

  {
    // run point to point resharding on non-disjoint meshes
    DeviceList first_mesh{{.start = 0, .num_devices = kTaskNumDevices}};
    DeviceList second_mesh{{.start = kTaskNumDevices - kTaskNumDevices / 2,
                            .num_devices = kTaskNumDevices}};
    test(first_mesh, second_mesh);
  }
}

int main(int argc, char** argv) {
  return zuku::program([] { return run(); }, argc, argv);
}
