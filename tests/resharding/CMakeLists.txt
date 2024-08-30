add_subdirectory(m-n-1d-resharding)
add_subdirectory(scatter-fully-sharded)
add_subdirectory(gather-fully-sharded)
add_subdirectory(deferred-delete)

add_library(reshard_test reshard_test.cc)
target_include_directories(reshard_test PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(reshard_test PRIVATE zuku::zuku shape_utils)

add_exe_test(partial-to-replicated
  N 4
  ARGV
   -level zuku-reshard=1
   -logfile test.log
  SOURCES
    partial-to-replicated.cc
  DEPENDENCIES
    reshard_test
)
