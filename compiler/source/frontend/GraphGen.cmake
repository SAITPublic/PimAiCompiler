target_include_directories(frontend PUBLIC
        ${GRAPHGEN_DIR}/core
        ${GRAPHGEN_DIR}/core/export
        ${GRAPHGEN_DIR}/core/graphgen_network
        ${GRAPHGEN_DIR}/core/import
        ${GRAPHGEN_DIR}/core/intermediate_process
        ${GRAPHGEN_DIR}/core/misc
        ${GRAPHGEN_DIR}/third_party/nlohmann)

foreach(libname graphgen_core graphgen_export graphgen_network
        graphgen_import graphgen_intermediate_process ir_builder plugin_chain graphgen_misc glog)
        find_library(lib${libname}
                    NAMES ${libname}
                    HINTS ${GRAPHGEN_BUILD_DIR}
                    PATH_SUFFIXES core core/export core/graphgen_network core/import
                                  core/intermediate_process core/misc third_party/glog/glog-build)
        target_link_libraries(frontend ${lib${libname}})
endforeach()
