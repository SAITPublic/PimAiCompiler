set(CMAKE_CXX_COMPILER "${ROCM_PATH}/bin/hipcc")
#set(HCC_CXX_FLAGS  "--amdgpu-target=gfx906")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${HCC_CXX_FLAGS}")

include_directories("${PROJECT_SOURCE_DIR}/runtime/include")

# include torch/csrc
include_directories(
    "${LIBTORCH_DIR}/include/torch/csrc"
    "${LIBTORCH_DIR}/include/torch/csrc/api"
    "${LIBTORCH_DIR}/include/torch/csrc/api/include"
    "${LIBTORCH_DIR}/include/torch/csrc/include"
)

add_executable(simpleMain simpleMain.cpp)

target_link_libraries(simpleMain nn_runtime torch_cpu)

