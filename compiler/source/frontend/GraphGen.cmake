target_include_directories(frontend PUBLIC
        ${GRAPHGEN_DIR}/core
        ${GRAPHGEN_DIR}/core/export
        ${GRAPHGEN_DIR}/core/graphgen_network
        ${GRAPHGEN_DIR}/core/import
        ${GRAPHGEN_DIR}/core/intermediate_process
        ${GRAPHGEN_DIR}/core/misc
        ${GRAPHGEN_DIR}/third_party/nlohmann)

target_link_directories(frontend PUBLIC 
        ${GRAPHGEN_BUILD_DIR}/core
        ${GRAPHGEN_BUILD_DIR}/core/export
        ${GRAPHGEN_BUILD_DIR}/core/graphgen_network
        ${GRAPHGEN_BUILD_DIR}/core/import
        ${GRAPHGEN_BUILD_DIR}/core/intermediate_process
        ${GRAPHGEN_BUILD_DIR}/core/misc
        ${GRAPHGEN_BUILD_DIR}/third_party/glog/glog-build)

target_link_libraries(frontend graphgen_core
        graphgen_export
        graphgen_network
        graphgen_import
        graphgen_intermediate_process
        ir_builder
        plugin_chain
        graphgen_misc
        glog)
