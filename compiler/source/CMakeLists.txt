add_subdirectory(frontend)
add_subdirectory(middlend)
# add_subdirectory(backend)

include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/compiler/include)

set(CMAKE_CXX_FLAGS "-std=c++17 -fPIC")
file(GLOB_RECURSE nncompiler_src *.cpp)
add_library(nncompiler "STATIC" ${nncompiler_src})

target_link_libraries(nncompiler common)
target_link_libraries(nncompiler ir)
target_link_libraries(nncompiler frontend)
target_link_libraries(nncompiler middlend)
# target_link_libraries(nncompiler backend)
