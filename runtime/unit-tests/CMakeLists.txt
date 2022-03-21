set(CMAKE_CXX_COMPILER "${ROCM_PATH}/bin/hipcc")
set(HCC_CXX_FLAGS  "--amdgpu-target=gfx906")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${HCC_CXX_FLAGS}")

option(ATEN_OP_UTEST "Build AtenOp UT" ON)

set(UT_LIBS gtest_main gtest nn_runtime torch torch_cpu c10 torch_hip c10_hip PimRuntime)

include_directories("${PROJECT_SOURCE_DIR}/runtime/include")

if(ATEN_OP_UTEST)
add_executable(NnrtAtenOpUnitTest "aten_ops_ut.cpp")
target_link_libraries(NnrtAtenOpUnitTest ${UT_LIBS})
endif()