set(CMAKE_CXX_FLAGS "-std=c++17 -fPIC")
file(GLOB_RECURSE runtime_builder_src *.cpp)
add_library(runtime_builder SHARED ${runtime_builder_src})

target_include_directories(runtime_builder PUBLIC
    ${PROJECT_SOURCE_DIR}      
    ${PROJECT_SOURCE_DIR}/runtime/include)

target_link_libraries(runtime_builder common new_ir hsakmt torch c10)

install (TARGETS runtime_builder LIBRARY DESTINATION ${ROCM_PATH}/lib/)
