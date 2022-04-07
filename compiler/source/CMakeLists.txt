include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/compiler/include)

set(CMAKE_CXX_FLAGS "-std=c++17 -fPIC")
file(GLOB_RECURSE nncompiler_src *.cpp)
add_library(nncompiler SHARED ${nncompiler_src})

target_link_libraries(nncompiler ir torch torch_cpu c10 -lstdc++fs)

install (TARGETS nncompiler LIBRARY DESTINATION ${ROCM_PATH}/lib/)
