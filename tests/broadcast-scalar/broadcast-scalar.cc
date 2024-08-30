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

ShardedShape GetShape(DeviceList devices) {
  ShardedShape shape{.type = zuku::SupportedType::F32,
                     .sharding = {.dims = {ShardingDim{
                                               .size = 1,
                                               .sharding = 1,
                                               .permutation = 0,
                                           },
                                           ShardingDim{
                                               .size = 1,
                                               .sharding = devices.size(),
                                               .permutation = 1,
                                           }},
                                  .devices = std::move(devices)}};
  return shape;
}

constexpr int64_t kNumIterations = 3;

auto test(DeviceList source_mesh, DeviceList target_mesh) {
  auto waiter = on(ProcessorGroup::Local()).defer([=](Processor p) {
    Store<ShardedArray> source =
        ShardedArray::Create(GetShape(source_mesh), {.processor = p});
    Store<ShardedArray> dest =
        ShardedArray::Create(GetShape(target_mesh), {.processor = p});

    if (dest->HasTile()) {
      log_test.debug() << "destination has instance " << dest->tile().instance()
                       << " on " << p.local_id();
    }
    if (source->HasTile()) {
      log_test.debug() << "source has instance " << source->tile().instance()
                       << " on " << p.local_id();
    }

    for (int64_t iter = 0; iter < kNumIterations; ++iter) {
      across(source_mesh)
          .if_on(p)
          .defer(
              [=](ShardedArray& array) {
                int* data = array.tile().ptr<int>();
                *data = iter + 1;
              },
              source);

      Reshard(p, source, dest);

      auto dest_check = across(target_mesh)
                            .if_on(p)
                            .defer(
                                [=](const ShardedArray& array) {
                                  const int* data = array.tile().ptr<int>();
                                  const int expected = iter + 1;
                                  if (*data != expected) {
                                    std::cerr
                                        << "unexpected value in dest data on "
                                        << p.global_id() << ": " << *data
                                        << " != " << expected << std::endl;
                                    abort();
                                  }
                                },
                                dest);

      across(target_mesh)
          .if_on(p)
          .defer(
              [=](ShardedArray& array) {
                int* data = array.tile().ptr<int>();
                *data = 10 * iter + 1;
              },
              dest);

      Reshard(p, dest, source);

      auto src_check = across(source_mesh)
                           .if_on(p)
                           .defer(
                               [=](const ShardedArray& array) {
                                 const int* data = array.tile().ptr<int>();
                                 const int expected = 10 * iter + 1;
                                 if (*data != expected) {
                                   std::cerr
                                       << "unexpected value in source data on "
                                       << p.global_id() << ": " << *data
                                       << " != " << expected << std::endl;
                                   abort();
                                 }
                               },
                               source);
      after(src_check).defer([=, &logger = log_test] {
        logger.debug() << "finished iteration " << iter << " on "
                       << p.global_id() << " for " << source_mesh << "->"
                       << target_mesh;
      });
    }
    source.Wait();
    dest.Wait();
  });
  waiter.Wait();
}

template <int NS, int NT, bool DJ>
constexpr std::tuple<int, int, bool> Test() {
  return {NS, NT, DJ};
}

auto run() {
  constexpr std::array kConfigs = {Test<3, 10, true>(), Test<3, 10, false>(),
                                   Test<4, 8, false>()};

  for (const auto& [num_source, num_dest, disjoint] : kConfigs) {
    DeviceList source_mesh{{.start = 0, .num_devices = num_source}};
    const int start = disjoint ? num_source : 0;
    DeviceList large_target_mesh{{.start = start, .num_devices = num_dest}};
    test(source_mesh, large_target_mesh);
  }
}

int main(int argc, char** argv) {
  return zuku::program([] { return run(); }, argc, argv);
}
