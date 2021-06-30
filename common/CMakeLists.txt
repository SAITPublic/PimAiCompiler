include_directories(${PROJECT_SOURCE_DIR})

set(CMAKE_CXX_FLAGS "-std=c++17 -fPIC")
file(GLOB_RECURSE common_src *.cpp *.h *.hpp)
add_library(common "STATIC" ${common_src})
