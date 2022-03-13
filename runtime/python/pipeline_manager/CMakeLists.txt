
include_directories(
    "${LIBTORCH_DIR}/include/torch/csrc"
    "${LIBTORCH_DIR}/include/torch/csrc/api"
    "${LIBTORCH_DIR}/include/torch/csrc/api/include"
    "${LIBTORCH_DIR}/include/torch/csrc/include"
)
file(GLOB pipeline_manager_py_source *.cpp)

add_library(pipeline_manager_py SHARED ${pipeline_manager_py_source})

target_include_directories(pipeline_manager_py PUBLIC
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/common
    ${PROJECT_SOURCE_DIR}/compiler/include
    ${PROJECT_SOURCE_DIR}/ir/include      
    ${PROJECT_SOURCE_DIR}/runtime/include)

target_link_libraries(pipeline_manager_py PUBLIC nncompiler common nn_runtime  torch_cpu torch)

install (TARGETS pipeline_manager_py LIBRARY DESTINATION ${ROCM_PATH}/lib/)
