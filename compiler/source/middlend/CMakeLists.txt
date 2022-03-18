include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/compiler/include/middlend)

set(CMAKE_CXX_FLAGS "-std=c++17 -fPIC")
file(GLOB_RECURSE middlend_src *.cpp)
add_library(middlend SHARED ${middlend_src})

target_link_libraries(middlend ir)
target_link_libraries(middlend -lstdc++fs)

install (TARGETS middlend LIBRARY DESTINATION ${ROCM_PATH}/lib/)
