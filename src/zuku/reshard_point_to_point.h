/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef _POC_SRC_RESHARD_POINT_TO_POINT_H_
#define _POC_SRC_RESHARD_POINT_TO_POINT_H_

#include "realm/event.h"
#include "realm/processor.h"

#include <queue>

#include "store.h"
#include "tiled_array.h"

namespace zuku {

void ReshardPointToPoint(Processor p, zuku::View<zuku::ShardedArray> src,
                         zuku::Store<zuku::ShardedArray>& dst);

}

#endif
