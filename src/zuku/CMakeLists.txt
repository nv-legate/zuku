#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#=============================================================================

add_library(zuku SHARED
  init.cc
  processor.cc
  profile.cc
  reshard.cc
  reshard_manager.cc
  reshard_point_to_point.cc
  reshard_scatter_shards.cc
  reshard_gather_shards.cc
  reshard_scatter_gather.cc
  reshard_broadcast.cc
  task.cc
  tiled_array.cc
  stream.cc
)

target_link_libraries(zuku PUBLIC Legion::Realm)
set_target_properties(zuku PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(zuku PROPERTIES ENABLE_EXPORTS 1)

if (CUDAToolkit_FOUND AND TARGET CUDA::nvtx3)
  target_compile_definitions(zuku PRIVATE zuku_HAS_NVTX)
  target_link_libraries(zuku PRIVATE CUDA::nvtx3)
endif()

if (CUDAToolkit_FOUND AND TARGET CUDA::cuda_driver)
  target_compile_definitions(zuku PRIVATE zuku_HAS_CUDA)
  target_include_directories(zuku PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
endif()

target_include_directories(zuku
                           PUBLIC
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
                           $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
                           $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
                           $<INSTALL_INTERFACE:include>)

add_library(zuku::zuku ALIAS zuku)
