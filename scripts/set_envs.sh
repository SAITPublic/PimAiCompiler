#!/bin/bash

# set  GraphGen *.so to $LD_LIBRARY_PATH
export GRAPH_GEN_BUILD_DIR=/path/to/GraphGen/build
export LD_LIBRARY_PATH=$GRAPH_GEN_BUILD_DIR/third_part/glog/glog-build:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$GRAPH_GEN_BUILD_DIR/third_party/glog/glog-build:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$GRAPH_GEN_BUILD_DIR/core/export:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$GRAPH_GEN_BUILD_DIR/core/graphgen_network:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$GRAPH_GEN_BUILD_DIR/core/import:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$GRAPH_GEN_BUILD_DIR/core/intermediate_process:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$GRAPH_GEN_BUILD_DIR/core/misc:$LD_LIBRARY_PATH
