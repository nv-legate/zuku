#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#=============================================================================

cmake_minimum_required(VERSION 3.22.1 FATAL_ERROR)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_C_FLAGS_ASAN "${CMAKE_C_FLAGS_RELWITHDEBINFO} -fsanitize=address -fno-omit-frame-pointer -fno-common" CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS_ASAN "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -fsanitize=address -fno-omit-frame-pointer -fno-common" CACHE STRING "" FORCE)
set(CMAKE_EXE_LINKER_FLAGS_ASAN "${CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO} -fsanitize=address" CACHE STRING "" FORCE)
set(CMAKE_SHARED_LINKER_FLAGS_ASAN "${CMAKE_SHARED_LINKER_FLAGS_RELWITHDEBINFO} -fsanitize=address" CACHE STRING "" FORCE)

set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo" "ASAN")

# ########################################################################################
# * Download and initialize RAPIDS CMake helpers -----------------------------

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/include_rapids.cmake)

include_rapids()

set(zuku_version 24.9.0)
project(zuku VERSION ${zuku_version} LANGUAGES C CXX)

include(cmake/cpm_utils.cmake)
include(cmake/third_party/get_realm.cmake)

find_package(CUDAToolkit)

add_subdirectory(src)

enable_testing()
add_subdirectory(tests)

include(CPack)
include(GNUInstallDirs)
rapids_cmake_install_lib_dir(lib_dir)

install(TARGETS zuku
        EXPORT zuku-exports
        FILE_SET zuku_headers)


rapids_export(
  INSTALL zuku
  EXPORT_SET zuku-exports
  GLOBAL_TARGETS zuku
  NAMESPACE zuku::
  LANGUAGES ${ENABLED_LANGUAGES}
)

# build export targets
rapids_export(
  BUILD zuku
  EXPORT_SET zuku-exports
  GLOBAL_TARGETS zuku
  NAMESPACE zuku::
  LANGUAGES ${ENABLED_LANGUAES}
)

