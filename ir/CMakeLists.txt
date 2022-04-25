include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/ir/include)

set(CMAKE_CXX_FLAGS "-std=c++17 -fPIC")
file(GLOB_RECURSE ir_src *.cpp)
add_library(ir SHARED ${ir_src})

target_include_directories("ir" SYSTEM PUBLIC ${PROJECT_SOURCE_DIR}/third_party/half/include)

target_link_libraries(ir PUBLIC torch c10 glog)

install (TARGETS ir LIBRARY DESTINATION ${ROCM_PATH}/lib/)
