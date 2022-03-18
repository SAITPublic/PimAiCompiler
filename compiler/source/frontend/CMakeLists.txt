set(CMAKE_CXX_FLAGS "-std=c++17 -fPIC")
file(GLOB_RECURSE frontend_src *.cpp)
add_library(frontend SHARED ${frontend_src})

target_include_directories(frontend PUBLIC
    ${PROJECT_SOURCE_DIR}      
    ${PROJECT_SOURCE_DIR}/compiler/include/frontend
    ${TORCH_INCLUDE_DIRS})

target_link_libraries(frontend ir)

install (TARGETS frontend LIBRARY DESTINATION ${ROCM_PATH}/lib/)
