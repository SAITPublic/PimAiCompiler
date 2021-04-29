get_filename_component(COMPILER_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../" REALPATH)
get_filename_component(SCHEMA_DEFAULT_DIR "${COMPILER_ROOT_DIR}/../npu_ir/" REALPATH)

if(NOT DEFINED SCHEMA_DIR)
    set(SCHEMA_DIR ${SCHEMA_DEFAULT_DIR})
endif()
get_filename_component(SCHEMA_DIR "${SCHEMA_DIR}" REALPATH)
if(NOT EXISTS ${SCHEMA_DIR} OR NOT IS_DIRECTORY "${SCHEMA_DIR}")
    message(FATAL_ERROR
        "IR Schema Directory does not exist.(${SCHEMA_DIR})\n"
        "Please specify with -DSCHEMA_DIR.\n"
        "If not set, default is ${SCHEMA_DEFAULT_DIR}")
endif()

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE BOTH)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt.in flatbuffers-download/CMakeLists.txt)
execute_process(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
                WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-download"
)
execute_process(COMMAND "${CMAKE_COMMAND}" --build .
                WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-download"
)
add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-src
                 ${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-build
                 EXCLUDE_FROM_ALL)
set(FLATBUFFERS_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-src/include PARENT_SCOPE)
set(FLATC_GENERATOR_DEPENDENCY flatc)

add_custom_target(flatc_python_generator ALL
                  COMMAND flatc -o ${PROJECT_BINARY_DIR}/python -p -I ${SCHEMA_DIR}/npu_schema --gen-all --gen-object-api ${SCHEMA_DIR}/ir.fbs
                  DEPENDS ${FLATC_GENERATOR_DEPENDENCY})
set_target_properties(flatc_python_generator
    PROPERTIES ADDITIONAL_CLEAN_FILES ${PROJECT_BINARY_DIR}/python)
