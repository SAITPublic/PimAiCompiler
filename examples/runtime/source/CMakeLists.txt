set(CMAKE_CXX_FLAGS "-std=c++17 -fPIC")

include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/examples/runtime/include)
include_directories(${PROJECT_SOURCE_DIR}/runtime/include)

# include torch/csrc
include_directories(
    "${LIBTORCH_DIR}/include/torch/csrc"
    "${LIBTORCH_DIR}/include/torch/csrc/api"
    "${LIBTORCH_DIR}/include/torch/csrc/api/include"
    "${LIBTORCH_DIR}/include/torch/csrc/include"
)

file(GLOB_RECURSE pipeline_manager_src *.cpp)
add_library(pipeline_manager "SHARED" ${pipeline_manager_src})

target_link_libraries(pipeline_manager nncompiler nn_runtime torch_cpu)
