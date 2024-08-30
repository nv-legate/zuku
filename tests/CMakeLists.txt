#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#=============================================================================

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.13.0.zip
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
include(GoogleTest)

function(add_exe_test name)
 cmake_parse_arguments(
   TEST
   "MPI_ONLY;GTEST"
   "N"
   "SOURCES;ARGV;DEPENDENCIES"
   ${ARGN}
 )

 execute_process(
    COMMAND whoami
    OUTPUT_VARIABLE username
 )
 string(STRIP ${username} username)
 if (username STREQUAL "root")
   set(ALLOW_ROOT_FLAG "--allow-run-as-root")
 endif()

 if ("${TEST_N}" STREQUAL "")
   message(FATAL_ERROR "Test ${name} did not specify the number of procs")
 endif()

 add_executable(${name} ${TEST_SOURCES})
 target_link_libraries(${name} PRIVATE zuku::zuku)
 target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_FUNCTION_LIST_DIR})

 if (TEST_DEPENDENCIES)
  target_link_libraries(${name} PRIVATE ${TEST_DEPENDENCIES})
 endif()

 if (TEST_N GREATER 1 AND MPI_FOUND)
  set(ARGV ${TEST_ARGV} -ll:cpu 1)
  add_test(NAME "MPI_${name}" COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${TEST_N} ${ALLOW_ROOT_FLAG} $<TARGET_FILE:${name}> ${ARGV})
  set_tests_properties("MPI_${name}" PROPERTIES ENVIRONMENT "UCX_TLS=^mm")
 endif()
 
 if (NOT TEST_MPI_ONLY)
  set(ARGV ${TEST_ARGV} -ll:cpu ${TEST_N})
  if (TEST_GTEST)
    gtest_discover_tests(${name}
        DISCOVERY_TIMEOUT 10
        DISCOVERY_MODE PRE_TEST)
  else()
    add_test(NAME ${name} COMMAND ${name} ${ARGV})
    set_tests_properties(${name} PROPERTIES TIMEOUT 10)
  endif()
 endif()
endfunction(add_exe_test)

function(add_unit_test name)
  add_exe_test(${name} GTEST N 1 ${ARGN})
  target_link_libraries(${name} PRIVATE GTest::gtest GTest::gmock)
  target_compile_features(${name} PUBLIC cxx_std_20)
  target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_FUNCTION_LIST_DIR})
endfunction(add_unit_test)

find_package(MPI)

add_subdirectory(hello-world)
add_subdirectory(vector-hello-world)
add_subdirectory(store-hello-world)
add_subdirectory(profiling-hello-world)
add_subdirectory(sharded-array)
add_subdirectory(vector-store)
add_subdirectory(broadcast-scalar)
add_subdirectory(unit)
add_subdirectory(resharding)

add_library(shape_utils INTERFACE shape_utils.h)
target_include_directories(shape_utils INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(shape_utils INTERFACE zuku::zuku)

if (CUDAToolkit_FOUND AND TARGET CUDA::cublas)
  add_subdirectory(control-events)
endif()
