/*
 * Copyright (C) 2019 Samsung Electronics Co. LTD
 *
 * This software is proprietary of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#include "ir/ir_importer.hpp"
#include "ir/common/log.hpp"
#include "ir/ir_blob_builder.hpp"
#include "ir/ir_graph_builder.hpp"
#include "ir/ir_includes.hpp"

namespace nn_compiler {

/**
 * @brief.      Create nn_ir::NNIR class from IR file.
 * @details.    This function parses IR file (flatbuffer) and
 *              instantiate an object of nn_ir::NNIR class
 * @param[in].  file_path Input file path
 * @param[out]. graphs NNIR graph list
 * @returns.    nn_ir::NNIR
 */
RetVal IRImporter::getNNIRFromFile(const std::string& file_path, std::vector<std::unique_ptr<nn_ir::NNIR>>& graphs) {
    Log::IR::I() << "IRImporter::getNNIRFromFile() is called, path: " << file_path;

    if (file_path.find("frontend.ir") != std::string::npos) {
        Log::IR::I() << "IRImporter::getNNIRFromFile() : Frontend IR";
    } else if (file_path.find("middle_end.ir") != std::string::npos) {
        Log::IR::I() << "IRImporter::getNNIRFromFile() : Middleend IR";
    }

    std::unique_ptr<char[]> data(nullptr);
    std::ifstream           infile;
    infile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    // Open IR File
    try {
        infile.open(file_path.c_str(), std::ios::binary | std::ios::in);
        infile.seekg(0, std::ios::end);
        int length = infile.tellg();
        infile.seekg(0, std::ios::beg);
        data.reset(new char[length]);
        infile.read(data.get(), length);
        infile.close();
    }
    catch (const std::ifstream::failure& e) {
        Log::IR::E() << "IRImporter::getNNIRFromFile() : FileOpenFail, path: " << file_path;
        return RetVal::FAILURE;
    }
    return buildNNIRFromData(data.get(), graphs);
}

RetVal IRImporter::buildNNIRFromData(const char* data, std::vector<std::unique_ptr<nn_ir::NNIR>>& graphs) {
    Log::IR::I() << "IRBuilder::buildNNIRFromData() is called";

    // Build graphs in IR
    auto           ir_root   = IR::GetRoot(data);
    auto           ir_graphs = ir_root->graphs();
    IRGraphBuilder ir_graph_builder;

    for (auto ir_graph : *ir_graphs) {
        auto graph = std::make_unique<nn_ir::NNIR>(ir_graph->id(), ir_graph->name()->str());

        // Parse blobs
        if (auto ir_blobs = ir_graph->blobs()) {
            IRBlobBuilder ir_blob_builder;
            for (auto ir_blob : *ir_blobs) {
                ir_blob_builder.getOrCreateBlob(ir_blob, *graph);
            }
        }

        // Parse nodes
        if (auto ir_nodes = ir_graph->nodes()) {
            for (auto ir_node : *ir_nodes) {
                auto node = ir_graph_builder.createNode(ir_node, *graph);
                graph->addNode(std::move(node));
            }
        }

        // Edges can contain blobs' MemoryInfo, so parse them after blobs
        if (auto ir_edges = ir_graph->edges()) {
            for (auto ir_edge : *ir_edges) {
                EDGE_ID_T id   = ir_edge->id();
                auto      edge = ir_graph_builder.createEdge(ir_edge, *graph);
                graph->addEdge({id, std::move(edge)});
            }
        }

        // Parse execution steps
        if (ir_graph->hw_info()) {
            if (auto ir_steps = ir_graph->hw_info()->execution_order()) {
                for (auto ir_step : *ir_steps) {
                    graph->addExecutionStep(ir_step);
                }
            }
        }

        graphs.push_back(std::move(graph));
    }
    return RetVal::SUCCESS;
}
} // namespace nn_compiler
