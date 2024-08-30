#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#=============================================================================

add_exe_test(
  control-events
  SOURCES
    control-events.cc
  N 1
)
target_link_libraries(control-events PRIVATE CUDA::cublas CUDA::cuda_driver CUDA::cudart)
