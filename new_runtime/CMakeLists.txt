option(BUILD_PY_RUNTIME "option for build py_nnruntime extension" ON)

set(CMAKE_CXX_FLAGS "-std=c++17 -fPIC")
add_subdirectory(source)

include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/new_runtime/include)

if(BUILD_PY_RUNTIME)
    add_subdirectory(python)
endif(BUILD_PY_RUNTIME)
