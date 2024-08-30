#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#=============================================================================

add_exe_test(deferred-delete
  MPI_ONLY
  N 4
  SOURCES
    deferred-delete.cc
  ARGV
    -ll:csize 1
    -level zuku-tile=1
    -level inst=1
    -logfile test.log
  DEPENDENCIES
    reshard_test
)
