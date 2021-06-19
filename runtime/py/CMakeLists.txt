# for python library
cmake_minimum_required(VERSION 3.4...3.18)

add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/pybind11 pybind11_binary_dir)
add_library(Nnr MODULE nn_runtime_py.cpp)
target_link_libraries(Nnr PRIVATE nn_runtime pybind11::module pybind11::lto)
pybind11_extension(Nnr)
pybind11_strip(Nnr)

install ( TARGETS Nnr LIBRARY DESTINATION ${ROCM_PATH}/lib/)


