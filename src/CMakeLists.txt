#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#=============================================================================

add_subdirectory(zuku)

target_sources(
  zuku
  PUBLIC
    FILE_SET zuku_headers
    TYPE HEADERS
    FILES
        zuku/barrier.h
        zuku/defer.h
        zuku/future.h
        zuku/hash.h
        zuku/init.h
        zuku/instance_cache.h
        zuku/link_promise.h
        zuku/mesh.h
        zuku/processor.h
        zuku/profile.h
        zuku/promise.h
        zuku/realm_variants.h
        zuku/reshard.h
        zuku/reshard_broadcast.h
        zuku/reshard_gather_shards.h
        zuku/reshard_manager.h
        zuku/reshard_point_to_point.h
        zuku/reshard_scatter_gather.h
        zuku/reshard_scatter_shards.h
        zuku/resolve_dependency.h
        zuku/shape.h
        zuku/store.h
        zuku/stream.h
        zuku/task.h
        zuku/task_base.h
        zuku/tiled_array.h
        zuku/token.h
        zuku/type_traits.h
        zuku/vector.h
        zuku/zuku_utils.h
)

find_library(LIBDL dl)
if(LIBDL)
  target_link_libraries(zuku PUBLIC ${LIBDL})
else()
  MESSAGE(FATAL_ERROR "unable to locate libdl")
endif()
