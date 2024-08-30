#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#=============================================================================

add_exe_test(broadcast-scalar
  N 13 # 3 source, 10 target
  SOURCES
    broadcast-scalar.cc
)
