#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#=============================================================================

add_unit_test(
  reshard_broadcast_test
  SOURCES
    reshard_broadcast_test.cc
)

add_unit_test(
  reshard_scatter_gather_test
  SOURCES
    reshard_scatter_gather_test.cc
)

add_unit_test(
  tiled_array_test
  SOURCES
    tiled_array_test.cc
)

add_unit_test(
  type_traits_test
  SOURCES
    type_traits_test.cc
)

add_unit_test(
  mem_alloc_failure_test
  SOURCES
    mem-alloc-failure.cc
)

add_unit_test(
  reshard_config_test
  SOURCES
    reshard_config_test.cc
  DEPENDENCIES
    reshard_test
)
