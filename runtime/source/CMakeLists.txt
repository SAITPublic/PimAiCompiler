file(GLOB_RECURSE runtime_source nn_runtime.cpp *.cpp)
set(CMAKE_CXX_COMPILER "${ROCM_PATH}/bin/hipcc")
set(HIPCC_FLAGS "--amdgpu-target=gfx906 --amdgpu-target=gfx908")
set(CXX_FLAGS "-std=c++17 -fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  ${HIPCC_FLAGS}")

include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/runtime/include)

add_library(nn_runtime SHARED ${runtime_source})
target_link_libraries(nn_runtime ir -lstdc++fs PimRuntime)

install (TARGETS nn_runtime LIBRARY DESTINATION ${ROCM_PATH}/lib/)
install (FILES ${CMAKE_CURRENT_SOURCE_DIR}/../include/nn_runtime.h DESTINATION ${ROCM_PATH}/include)
