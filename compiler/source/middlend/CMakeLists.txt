include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/compiler/include/middlend)

file(GLOB_RECURSE middlend_src *.cpp ${PROJECT_SOURCE_DIR}/third_party/json/*.cpp)
add_executable("middlend" ${middlend_src})

target_link_libraries("middlend" common)
target_link_libraries("middlend" ir)

install(TARGETS "middlend" RUNTIME DESTINATION ${CMAKE_BINARY_DIR}/compiler/middlend)
