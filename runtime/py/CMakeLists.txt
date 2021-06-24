# for python library
cmake_minimum_required(VERSION 3.4...3.18)

add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/pybind11 pybind11_binary_dir)
add_library(Nnrt MODULE nn_runtime_py.cpp)
target_link_libraries(Nnrt PRIVATE nn_runtime pybind11::module pybind11::lto)
pybind11_extension(Nnrt)
pybind11_strip(Nnrt)

install ( TARGETS Nnrt LIBRARY DESTINATION ${ROCM_PATH}/lib/)


