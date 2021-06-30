include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_BINARY_DIR}/third_party/flatbuffers/flatbuffers-src/include)

set(CMAKE_CXX_FLAGS "-std=c++17 -fPIC")
file(GLOB_RECURSE ir_src *.cpp)
add_library(ir "STATIC" ${ir_src})

target_include_directories("ir" SYSTEM PUBLIC ${PROJECT_SOURCE_DIR}/third_party/half/include)

target_link_libraries(ir common)

file(GLOB FLATC_GENERATED_FILES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/include/generated/*.h)
list(LENGTH FLATC_GENERATED_FILES FLATC_FILE_LEN)

if (FLATC_FILE_LEN EQUAL 0)
    add_dependencies(ir flatc_cpp_generator)
endif()
