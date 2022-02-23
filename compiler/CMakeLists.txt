add_subdirectory(source)

include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/compiler/include)

file(GLOB main_src *.cpp)
add_executable("compiler" ${main_src})

target_link_libraries("compiler" nncompiler)
