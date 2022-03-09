
include_directories(
    "${LIBTORCH_DIR}/include/torch/csrc"
    "${LIBTORCH_DIR}/include/torch/csrc/api"
    "${LIBTORCH_DIR}/include/torch/csrc/api/include"
    "${LIBTORCH_DIR}/include/torch/csrc/include"
)
file(GLOB new_pileline_manager_source *.cpp)

add_library(new_pileline_manager SHARED ${new_pileline_manager_source})
add_dependencies(new_pileline_manager flatc_cpp_generator)

target_include_directories(new_pileline_manager PUBLIC
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/common
    ${PROJECT_SOURCE_DIR}/compiler/include
    ${PROJECT_SOURCE_DIR}/new_ir/include      
    ${PROJECT_SOURCE_DIR}/new_runtime/include)

target_link_libraries(new_pileline_manager PUBLIC nncompiler common new_nn_runtime  torch_cpu torch)

install (TARGETS new_pileline_manager LIBRARY DESTINATION ${ROCM_PATH}/lib/)
