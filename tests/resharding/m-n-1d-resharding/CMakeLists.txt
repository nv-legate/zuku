#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#=============================================================================

add_exe_test(m-n-1d-resharding
  N 6
  SOURCES
    m-n-1d-resharding.cc
  DEPENDENCIES
    reshard_test
)
