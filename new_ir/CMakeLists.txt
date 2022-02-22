include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/new_ir/include)

set(CMAKE_CXX_COMPILER "${ROCM_PATH}/bin/hipcc")
set(CMAKE_CXX_FLAGS "-std=c++17 -fPIC")
file(GLOB_RECURSE new_ir_src *.cpp)
add_library(new_ir SHARED ${new_ir_src})

target_include_directories("new_ir" SYSTEM PUBLIC ${PROJECT_SOURCE_DIR}/third_party/half/include)

target_link_libraries(new_ir PUBLIC common torch c10 glog)

install (TARGETS new_ir LIBRARY DESTINATION ${ROCM_PATH}/lib/)
