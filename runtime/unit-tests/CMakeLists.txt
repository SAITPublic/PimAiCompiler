aux_source_directory(. nnr_ut_source)
#aux_source_directory(manager nnr_ut_source)
#aux_source_directory(executor nnr_ut_source)

set(CMAKE_CXX_COMPILER "${ROCM_PATH}/bin/hipcc")
set(HCC_CXX_FLAGS  "--amdgpu-target=gfx906")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${HCC_CXX_FLAGS}")

include_directories("${PROJECT_SOURCE_DIR}/runtime/include")
add_executable(NnrtUnitTest ${nnr_ut_source})

target_link_libraries(NnrtUnitTest gtest_main gtest nn_runtime)
