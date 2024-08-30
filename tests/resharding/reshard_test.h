/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ZUKU_TESTS_RESHARDING_RESHARD_TEST_H_
#define ZUKU_TESTS_RESHARDING_RESHARD_TEST_H_

#include "mesh.h"
#include "shape.h"

namespace zuku {

void run_reshard_test(ShardedShape source_shape, ShardedShape target_shape,
                      DeviceList source_mesh, DeviceList target_mesh,
                      int num_iterations);

}  // namespace zuku

#endif
