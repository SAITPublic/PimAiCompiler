aux_source_directory(. runtime_source)
aux_source_directory(manager runtime_source)
aux_source_directory(executor runtime_source)

set(CMAKE_CXX_COMPILER "${ROCM_PATH}/bin/hipcc")
set(HIPCC_FLAGS "--amdgpu-target=gfx906")
set(CXX_FLAGS "-std=c++11")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${HIPCC_FLAGS}")

add_library(nn_runtime SHARED ${runtime_source})

target_include_directories(nn_runtime PRIVATE ${TORCH_INCLUDE_DIR})
target_link_libraries (nn_runtime hsakmt glog ${TORCH_LIBRARIES})


install ( TARGETS nn_runtime LIBRARY DESTINATION ${ROCM_PATH}/lib/)
install ( FILES ${CMAKE_CURRENT_SOURCE_DIR}/../include/nn_runtime.h DESTINATION ${ROCM_PATH}/include )
#install ( FILES ${CMAKE_CURRENT_SOURCE_DIR}/../include/_data_types.h DESTINATION ${ROCM_PATH}/include )
