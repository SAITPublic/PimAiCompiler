set(CMAKE_CXX_COMPILER "${ROCM_PATH}/bin/hipcc")
set(HCC_CXX_FLAGS  "--amdgpu-target=gfx906")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${HCC_CXX_FLAGS}")

option(ATEN_OP_UTEST "Build AtenOp UT" OFF)
option(PRIM_OP_UTEST "Build PrimOp UT" OFF)
option(CUSTOM_OP_UTEST "Build CustomOp UT" ON)

set(UT_LIBS gtest_main gtest nn_runtime torch torch_cpu c10 torch_hip c10_hip)

include_directories("${PROJECT_SOURCE_DIR}/runtime/include")
if(CUSTOM_OP_UTEST)
add_executable(NnrtCustomOpUnitTest "custom_op_ut.cpp")
target_link_libraries(NnrtCustomOpUnitTest ${UT_LIBS})
endif()

if(PRIM_OP_UTEST)
add_executable(NnrtPrimOpUnitTest "prim_op_ut.cpp")
target_link_libraries(NnrtPrimOpUnitTest ${UT_LIBS})
endif()

if(ATEN_OP_UTEST)
add_executable(NnrtAtenOpUnitTest "nn_op_ut.cpp")
target_link_libraries(NnrtAtenOpUnitTest ${UT_LIBS})
endif()
