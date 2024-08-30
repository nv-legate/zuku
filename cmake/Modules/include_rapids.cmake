#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#=============================================================================

macro(include_rapids)
  if (zuku_RAPIDS_DIR)
    include(${zuku_RAPIDS_DIR}/RAPIDS.cmake)
  else()
    if(NOT rapids-cmake-version)
      # default
      set(rapids-cmake-version 25.06)
      set(rapids-cmake-sha "365322aca32fd6ecd7027f5d7ec7be50b7f3cc2a")
    endif()
    if(NOT EXISTS ${CMAKE_BINARY_DIR}/RAPIDS.cmake)
      file(DOWNLOAD
           https://raw.githubusercontent.com/rapidsai/rapids-cmake/branch-${rapids-cmake-version}/RAPIDS.cmake
           ${CMAKE_BINARY_DIR}/RAPIDS.cmake)
    endif()
    include(${CMAKE_BINARY_DIR}/RAPIDS.cmake)
  endif()
  include(rapids-cmake)
  include(rapids-cpm)
  include(rapids-cuda)
  include(rapids-export)
  include(rapids-find)
endmacro()
