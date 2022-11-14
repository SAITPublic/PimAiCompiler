/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system, or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 * (Use of the Software is restricted to non-commercial, personal or academic, research purpose only)
 */

#include "frontend/importer/utils/utils.h"

namespace nn_compiler
{
namespace frontend
{
/*
 * Module--->SubModule--->SubModule.methods--->SubModules.method[i].graph
 * --->SubModules.method[i].graph.nodes
 */
#define SPLIT_NODE(i)             \
    DLOG(INFO) << "[" << i << "]" \
               << "===================Node=================="

#define SPLIT_GRAPH(i)            \
    DLOG(INFO) << "[" << i << "]" \
               << "===================Graph=================="

#define SPLIT(name) DLOG(INFO) << "*****" << name << "*****";

#define PRINT_INFO 1
int g_node_cnt = 0;

int TorchscriptPrinter::printGraphRecursive(std::string filename)
{
    torch::jit::Module script_model = torch::jit::load(filename);
    int ret = 0;
    for (auto m : script_model.named_modules()) {
        std::string module_name = m.name;
        torch::jit::Module submodule = m.value;
        DLOG(INFO) << "submodule_name:" << module_name;
        for (auto f : submodule.get_methods()) {
            SPLIT_GRAPH(ret + 1);
            DLOG(INFO) << "method_name:" << f.name();
            DLOG(INFO) << f.graph()->toString(true);
            ret++;
        }
    }
    return ret;
}

// Ref: pytorch_v180/torch/csrc/jit/ir/ir.cpp
void TorchscriptPrinter::printNodeRecursive(torch::jit::Node *node, size_t level)
{
    SPLIT_NODE(g_node_cnt++);
    // Get the node name
    std::string scope_name = node->scopeName();
    std::string node_name = node->kind().toQualString();  // like aten::mm, prim::Constant

#if PRINT_INFO
    DLOG(INFO) << "scope_name:" << scope_name;
    DLOG(INFO) << "node_name:" << node_name;
#endif

    // Get the inputs of node
    SPLIT("Inputs");
    int num_inputs = node->inputs().size();
    if (num_inputs == 0) DLOG(INFO) << "None";

    for (size_t i = 0; i < node->inputs().size(); i++) {
        // input type & name
        std::string input_type = node->inputs().at(i)->type()->str();
        std::string input_debug_name = node->inputs().at(i)->debugName();
        DLOG(INFO) << "input_type:" << input_type;
        DLOG(INFO) << "input_debug_name:%" << input_debug_name;
    }

    // Get the outputs of node
    SPLIT("Outputs");
    int num_outputs = node->outputs().size();
    if (num_outputs == 0) DLOG(INFO) << "None";
#if PRINT_INFO
    DLOG(INFO) << "num_outputs:" << num_outputs;
#endif

    for (size_t i = 0; i < node->outputs().size(); i++) {
        std::string output_type = node->outputs().at(i)->type()->str();
        std::string output_debug_name = node->outputs().at(i)->debugName();  // like: %1, %hidden.1
#if PRINT_INFO
        DLOG(INFO) << "output_type:" << output_type;
        DLOG(INFO) << "output_debug_name: %" << output_debug_name;
#endif
    }

    // Get the attrs of node
    SPLIT("Attrs")
    int num_attrs = node->numAttributes();
    if (num_attrs == 0) DLOG(INFO) << "None";
    DLOG(INFO) << "num_attrs:" << num_attrs;
    for (auto attr_name : node->attributeNamesS()) {
#if PRINT_INFO
        DLOG(INFO) << "attr_name:" << attr_name;
#endif
    }
    // the body of node
    // Recursive
    // Get each block of node
    for (size_t i = 0; i < node->blocks().size(); i++) {
        auto b = node->blocks()[i];
        // Get the nodes of each block
        for (auto nested : b->nodes()) {
            printNodeRecursive(nested, level + 2);
        }
    }
}

/**
 * @brief Get and Print All Nodes of torchScript model
 * @param filename file of torchScript model
 */
void TorchscriptPrinter::printScriptModelRecursive(std::string filename)
{
    torch::jit::Module script_model = torch::jit::load(filename);
    for (auto m : script_model.named_modules()) {
        // Get each sub-module
        std::string module_name = m.name;
        auto submodule = m.value;

        // Get Method of sub-module
        for (auto f : submodule.get_methods()) {
            // Get Graph of Method
            auto graph = f.function().graph();

            // Get the Inputs of Graph
            auto inputs = graph->inputs();
            DLOG(INFO) << "num_inputs_graph: " << inputs.size();
            // Get the type of each input
            for (size_t i = 0; i < inputs.size(); i++) {
                DLOG(INFO) << "type: " << inputs.at(i)->type()->str();
            }

            // Get the Outputs of Graph
            auto outputs = graph->outputs();
            DLOG(INFO) << "num_outputs_graph: " << outputs.size();
            // Get the type of each output
            for (size_t i = 0; i < outputs.size(); i++) {
                DLOG(INFO) << "type: " << *outputs.at(i)->type();
            }

            // Get Nodes of Graph
            for (auto node : graph->nodes()) {
                // Some Node may containes Blocks, each block contains nodes
                // there need to Get nodes recursively
                printNodeRecursive(node, 1);
            }
        }
    }
}

}  // namespace frontend
}  // namespace nn_compiler
