add_subdirectory(source)

include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/compiler/include)

file(GLOB_RECURSE main_srcc *.cpp)
add_executable("compiler" ${main_srcc})

target_link_libraries("compiler" nncompiler)
