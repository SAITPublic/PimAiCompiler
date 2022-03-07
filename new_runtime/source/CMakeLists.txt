add_subdirectory(builder)
add_subdirectory(executor)

file(GLOB_RECURSE new_runtime_source nn_runtime.cpp *.cpp)
set(CMAKE_CXX_COMPILER "${ROCM_PATH}/bin/hipcc")
set(HIPCC_FLAGS "--amdgpu-target=gfx906")
set(CXX_FLAGS "-std=c++17 -fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  ${HIPCC_FLAGS}")

include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/new_runtime/include)

add_library(new_nn_runtime SHARED ${new_runtime_source})
target_link_libraries(new_nn_runtime nncompiler runtime_builder PimRuntime)

install (TARGETS new_nn_runtime LIBRARY DESTINATION ${ROCM_PATH}/lib/)
install (FILES ${CMAKE_CURRENT_SOURCE_DIR}/../include/nn_runtime.h DESTINATION ${ROCM_PATH}/include)
