# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set(version "${Legion_major_version}.${Legion_minor_version}.${Legion_patch_version}")
set(exclude_from_all ${PKG_EXCLUDE_FROM_ALL})
set(FIND_PKG_ARGS
    GLOBAL_TARGETS Legion::Realm Legion::RealmRuntime
    BUILD_EXPORT_SET legate-core-exports 
    INSTALL_EXPORT_SET legate-core-exports)

if((NOT CPM_Legion_SOURCE) AND (NOT CPM_DOWNLOAD_Legion))
  # First try to find Legion via find_package() so the `Legion_USE_*` variables are
  # visible Use QUIET find by default.
  set(_find_mode QUIET)
  # If Legion_DIR/Legion_ROOT are defined as something other than empty or NOTFOUND use
  # a REQUIRED find so that the build does not silently download Legion.
  if(Legion_DIR OR Legion_ROOT)
    set(_find_mode REQUIRED)
  endif()
  rapids_find_package(Legion ${poc_version} EXACT CONFIG ${_find_mode} ${FIND_PKG_ARGS})
endif()

if(Legion_FOUND)
  message(STATUS "CPM: using local package Legion@${Legion_DIR}")
else()
  set(git_repo  "https://gitlab.com/StanfordLegion/legion.git")
  set(git branch "6da0710825fdb74e3a0f6363601d7fc6052c737d")
  include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/cpm_helpers.cmake)
  get_cpm_git_args(legion_cpm_git_args REPOSITORY ${git_repo} BRANCH ${git_branch})


  rapids_cpm_find(Legion ${poc_version} ${FIND_PKG_ARGS}
                  CPM_ARGS ${legion_cpm_git_args}
                           # HACK: Legion headers contain *many* warnings, but we would
                           # like to build with -Wall -Werror. But there is a
                           # work-around. Compilers treat system headers as special and
                           # do not emit any warnings about suspect code in them, so
                           # until legion cleans house, we mark their headers as
                           # "system" headers.
                           FIND_PACKAGE_ARGUMENTS
                           EXACT
                           SYSTEM
                           TRUE
                  EXCLUDE_FROM_ALL ${exclude_from_all}
                  PATCH_COMMAND ${patch_command} -d ./bindings/python --verbose
                  OPTIONS ${_legion_cuda_options}
                          "CMAKE_CXX_STANDARD 17"
                          "Legion_VERSION ${version}"
                          "Legion_BUILD_BINDINGS ON"
                          "Legion_REDOP_HALF ON"
                          "Legion_REDOP_COMPLEX ON"
                          "Legion_UCX_DYNAMIC_LOAD ON"
                          "CMAKE_SUPPRESS_DEVELOPER_WARNINGS ON")
endif()

set(Realm_HEADERS
realm.h
realm/activemsg.h
realm/activemsg.inl
realm/atomics.h
realm/atomics.inl
realm/bgwork.h
realm/bytearray.h
realm/bytearray.inl
realm/caching_allocator.h
realm/circ_queue.h
realm/circ_queue.inl
realm/cmdline.h
realm/cmdline.inl
realm/codedesc.h
realm/codedesc.inl
realm/compiler_support.h
realm/cuda/cuda_access.h
realm/cuda/cuda_access.inl
realm/cuda/cuda_internal.h
realm/cuda/cuda_memcpy.h
realm/cuda/cuda_module.h
realm/cuda/cuda_module.inl
realm/cuda/cuda_redop.h
realm/cuda/cudart_hijack.h
realm/custom_serdez.h
realm/custom_serdez.inl
realm/deppart/byfield.h
realm/deppart/deppart_config.h
realm/deppart/image.h
realm/deppart/inst_helper.h
realm/deppart/partitions.h
realm/deppart/partitions.inl
realm/deppart/preimage.h
realm/deppart/rectlist.h
realm/deppart/rectlist.inl
realm/deppart/setops.h
realm/deppart/sparsity_impl.h
realm/deppart/sparsity_impl.inl
realm/dynamic_table.h
realm/dynamic_table.inl
realm/dynamic_templates.h
realm/dynamic_templates.inl
realm/event.h
realm/event.inl
realm/event_impl.h
realm/event_impl.inl
realm/faults.h
realm/faults.inl
realm/id.h
realm/id.inl
realm/idx_impl.h
realm/indexspace.h
realm/indexspace.inl
realm/inst_impl.h
realm/inst_layout.h
realm/inst_layout.inl
realm/instance.h
realm/instance.inl
realm/interval_tree.h
realm/interval_tree.inl
realm/kokkos_interop.h
realm/lists.h
realm/lists.inl
realm/llvmjit/llvmjit.h
realm/llvmjit/llvmjit.inl
realm/llvmjit/llvmjit_internal.h
realm/llvmjit/llvmjit_module.h
realm/logging.h
realm/logging.inl
realm/machine.h
realm/machine.inl
realm/machine_impl.h
realm/mem_impl.h
realm/mem_impl.inl
realm/memory.h
realm/metadata.h
realm/module.h
realm/module_config.h
realm/module_config.inl
realm/mpi/am_mpi.h
realm/mpi/mpi_module.h
realm/mutex.h
realm/mutex.inl
realm/network.h
realm/network.inl
realm/nodeset.h
realm/nodeset.inl
realm/numa/numa_module.h
realm/numa/numasysif.h
realm/nvtx.h
realm/openmp/openmp_internal.h
realm/openmp/openmp_module.h
realm/openmp/openmp_threadpool.h
realm/openmp/openmp_threadpool.inl
realm/operation.h
realm/operation.inl
realm/point.h
realm/point.inl
realm/pri_queue.h
realm/pri_queue.inl
realm/proc_impl.h
realm/processor.h
realm/processor.inl
realm/procset/procset_module.h
realm/profiling.h
realm/profiling.inl
realm/python/python_internal.h
realm/python/python_module.h
realm/python/python_source.h
realm/python/python_source.inl
realm/realm_c.h
realm/realm_config.h
realm/redop.h
realm/repl_heap.h
realm/reservation.h
realm/reservation.inl
realm/rsrv_impl.h
realm/runtime.h
realm/runtime_impl.h
realm/sampling.h
realm/sampling.inl
realm/sampling_impl.h
realm/serialize.h
realm/serialize.inl
realm/shm.h
realm/sparsity.h
realm/sparsity.inl
realm/subgraph.h
realm/subgraph.inl
realm/subgraph_impl.h
realm/tasks.h
realm/threads.h
realm/threads.inl
realm/timers.h
realm/timers.inl
realm/transfer/address_list.h
realm/transfer/channel.h
realm/transfer/channel.inl
realm/transfer/channel_disk.h
realm/transfer/ib_memory.h
realm/transfer/lowlevel_dma.h
realm/transfer/transfer.h
realm/transfer/transfer.inl
realm/transfer/transfer_utils.h
realm/transfer/transfer_utils.inl
realm/ucx/bootstrap/bootstrap.h
realm/ucx/bootstrap/bootstrap_internal.h
realm/ucx/bootstrap/bootstrap_loader.h
realm/ucx/bootstrap/bootstrap_util.h
realm/ucx/mpool.h
realm/ucx/spinlock.h
realm/ucx/ucp_context.h
realm/ucx/ucp_dyn_load.h
realm/ucx/ucp_internal.h
realm/ucx/ucp_module.h
realm/ucx/ucp_utils.h
realm/utils.h
realm/utils.inl
realm_defines.h)

get_target_property(REALM_INCLUDE_DIRS Legion::RealmRuntime INTERFACE_INCLUDE_DIRECTORIES)



foreach(HEADER ${Realm_HEADERS})
  foreach(Base_DIR ${REALM_INCLUDE_DIRS})
      if (EXISTS "${Base_DIR}/${HEADER}")
        get_filename_component(HEADER_ROOT ${HEADER} DIRECTORY)
        set(COPY_DIR ${CMAKE_SOURCE_DIR}/realm/${HEADER_ROOT})
        file(MAKE_DIRECTORY ${COPY_DIR})
        file(COPY "${Base_DIR}/${HEADER}" DESTINATION ${COPY_DIR})
        break()
      endif()
  endforeach()
endforeach()


message("Have ${REALM_INCLUDE_DIRS}")
