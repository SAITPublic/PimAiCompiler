include_directories(${PROJECT_SOURCE_DIR})

file(GLOB_RECURSE common_src *.cpp *.h *.hpp)
add_library(common "STATIC" ${common_src})
