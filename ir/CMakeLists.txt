include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/ir/include)
include_directories(${PROJECT_BINARY_DIR}/third_party/flatbuffers/flatbuffers-src/include)

set(CMAKE_CXX_COMPILER "${ROCM_PATH}/bin/hipcc")
set(CMAKE_CXX_FLAGS "-std=c++17 -fPIC")
file(GLOB_RECURSE ir_src *.cpp)
add_library(ir SHARED ${ir_src})
add_dependencies(ir flatc_cpp_generator)

target_include_directories("ir" SYSTEM PUBLIC ${PROJECT_SOURCE_DIR}/third_party/half/include)

target_link_libraries(ir PUBLIC common torch c10)

install (TARGETS ir LIBRARY DESTINATION ${ROCM_PATH}/lib/)
