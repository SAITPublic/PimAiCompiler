# set(CMAKE_CXX_FLAGS "-std=c++17 -fPIC")
# file(GLOB_RECURSE stream_executor_src *.cpp)
# add_library(stream_executor SHARED ${stream_executor_src})

# target_include_directories(stream_executor PUBLIC
#     ${PROJECT_SOURCE_DIR}
#     ${PROJECT_SOURCE_DIR}/new_runtime/include)

# target_link_libraries(stream_executor common new_ir hsakmt torch c10)

# install (TARGETS stream_executor LIBRARY DESTINATION ${ROCM_PATH}/lib/)
