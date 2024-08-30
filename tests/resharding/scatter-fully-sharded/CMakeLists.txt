#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#=============================================================================

add_exe_test(scatter-fully-sharded
  N 12
  SOURCES
    scatter-fully-sharded.cc
  ARGV
    -level zuku-reshard=1
    -level unit-test=1
    -logfile test.log
  DEPENDENCIES
    reshard_test
)
