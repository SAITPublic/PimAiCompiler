get_filename_component(GRAPHGEN_DIR "${PROJECT_SOURCE_DIR}/../GraphGen/" REALPATH)
if(NOT EXISTS ${GRAPHGEN_DIR} OR NOT IS_DIRECTORY "${GRAPHGEN_DIR}")
    message(FATAL_ERROR
            "\n GraphGen Directory does not exist.(${GRAPHGEN_DIR}) \n")
endif()
set(PIM_IR_GENERATOR_PATH ${GRAPHGEN_DIR}/build/tools/pim_ir_generator)
if(NOT EXISTS ${PIM_IR_GENERATOR_PATH} OR NOT IS_DIRECTORY "${PIM_IR_GENERATOR_PATH}")
    message(FATAL_ERROR
            "\n Please build GraphGen. \n")
endif()

include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/compiler/include/frontend)

set(CMAKE_CXX_FLAGS "-std=c++17 -fPIC")
file(GLOB_RECURSE frontend_src *.cpp)
add_library(frontend "STATIC" ${frontend_src})

target_link_libraries(frontend common)
target_link_libraries(frontend ir)
