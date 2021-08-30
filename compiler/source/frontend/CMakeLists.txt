get_filename_component(GRAPHGEN_DIR "${PROJECT_SOURCE_DIR}/../GraphGen/" REALPATH)
if(NOT EXISTS ${GRAPHGEN_DIR} OR NOT IS_DIRECTORY "${GRAPHGEN_DIR}")
    message(FATAL_ERROR
            "\n GraphGen Directory does not exist.(${GRAPHGEN_DIR}) \n")
endif()
set(GRAPHGEN_BUILD_DIR ${GRAPHGEN_DIR}/build/)
if(NOT EXISTS ${GRAPHGEN_BUILD_DIR} OR NOT IS_DIRECTORY "${GRAPHGEN_BUILD_DIR}")
    message(FATAL_ERROR
            "\n Please build GraphGen. \n")
endif()

include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/compiler/include/frontend)

set(CMAKE_CXX_FLAGS "-std=c++17 -fPIC")
file(GLOB_RECURSE frontend_src *.cpp)
add_library(frontend "STATIC" ${frontend_src})

target_include_directories(frontend PUBLIC
        ${GRAPHGEN_DIR}/core
        ${GRAPHGEN_DIR}/core/export
        ${GRAPHGEN_DIR}/core/graphgen_network
        ${GRAPHGEN_DIR}/core/import
        ${GRAPHGEN_DIR}/core/intermediate_process
        ${GRAPHGEN_DIR}/core/misc
        ${GRAPHGEN_DIR}/third_party/nlohmann)

target_link_libraries(frontend graphgen_core graphgen_export graphgen_network
        graphgen_import graphgen_intermediate_process ir_builder plugin_chain
        graphgen_misc glog common ir)
