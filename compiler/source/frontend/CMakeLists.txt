include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/compiler/include/frontend)

set(CMAKE_CXX_FLAGS "-std=c++17 -fPIC")
file(GLOB_RECURSE frontend_src *.cpp)
add_library(frontend "STATIC" ${frontend_src})

target_link_libraries(frontend common ir new_ir)
