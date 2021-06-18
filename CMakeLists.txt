cmake_minimum_required(VERSION 3.5.1)
project(NNCompiler)

set(NN_COMPILER_VERSION_MAJOR 1)
set(NN_COMPILER_VERSION_MINOR 0)

set(PREFIX ${CMAKE_INSTALL_PREFIX})
set(EXEC_PREFIX "${PREFIX}/bin")
set(ROCM_PATH $ENV{ROCM_PATH})

# Set CMake Flags
set(CMAKE_CXX_FLAGS "-std=c++17 -Wall -Wno-unused-command-line-argument")
set(CMAKE_C_FLAGS "-std=c17")

#Define minimum version for GCC
set(GCC_EXPECTED_VERSION 4.9.0)
if(GCC_EXPECTED_VERSION VERSION_GREATER CMAKE_CXX_COMPILER_VERSION)
    message(FATAL_ERROR "GCC: NNCompiler-c++ requires atleast version ${GCC_EXPECTED_VERSION}")
endif()

include_directories("${PROJECT_SOURCE_DIR}/runtime/include"
                    "${PROJECT_SOURCE_DIR}/external_libs/include"
                    "${ROCM_PATH}/include")
link_directories("${ROCM_PATH}/hsa/lib" "${ROCM_PATH}/libhsakmt/lib" "${ROCM_PATH}/lib" "${PROJECT_SOURCE_DIR}/external_libs")

#for test vectors default data path
add_definitions(-DROCM_PATH="${ROCM_PATH}")

add_subdirectory(common)
add_subdirectory(ir)
add_subdirectory(runtime)
#add_subdirectory(compiler)
#add_subdirectory(python_libs)
#add_subdirectory(examples)
#add_subdirectory(external_libs)
add_subdirectory(third_party/flatbuffers)
add_dependencies(nn_runtime flatc_cpp_generator)
