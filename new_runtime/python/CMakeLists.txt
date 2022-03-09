# for python library
cmake_minimum_required(VERSION 3.4...3.18)

# fix torch c10::LOG and glog
add_definitions(-DC10_USE_GLOG)

# Fix pthread not found
# https://blog.csdn.net/hongge_smile/article/details/108789449
set(CMAKE_THREAD_LIBS_INIT "-lpthread")
set(CMAKE_HAVE_THREADS_LIBRARY 1)
set(CMAKE_USE_WIN32_THREADS_INIT 0)
set(CMAKE_USE_PTHREADS_INIT 1)
set(THREADS_PREFER_PTHREAD_FLAG ON)


# Add rocm dir
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH}
${LIBTORCH_DIR}
${ROCM_PATH}
"${ROCM_PATH}/hip"
)

# include torch/csrc
include_directories(
    "${LIBTORCH_DIR}/include/torch/csrc"
    "${LIBTORCH_DIR}/include/torch/csrc/api"
    "${LIBTORCH_DIR}/include/torch/csrc/include"
)

# find torch & rocm libs
find_package(Torch REQUIRED)
# https://rocmdocs.amd.com/en/latest/Installation_Guide/Using-CMake-with-AMD-ROCm.html?highlight=cmake
find_package(rocblas)
find_package(rocfft)
find_package(hipsparse)
find_package(hiprand)

add_subdirectory(pipeline_manager)

add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/pybind11 pybind11_binary_dir)
add_library(NNCompiler MODULE nn_runtime_py.cpp)


set(TORCH_INSTALL_PREFIX "${LIBTORCH_DIR}")
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")

# for debug
# message(STATUS ${TORCH_LIBRARIES})
# message(STATUS ${TORCH_PYTHON_LIBRARY})

target_include_directories(NNCompiler PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(NNCompiler PRIVATE new_pileline_manager pybind11::module pybind11::lto ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})
set_property(TARGET NNCompiler PROPERTY CXX_STANDARD 17)


pybind11_extension(NNCompiler)
pybind11_strip(NNCompiler)

install ( TARGETS NNCompiler LIBRARY DESTINATION ${ROCM_PATH}/lib/)
