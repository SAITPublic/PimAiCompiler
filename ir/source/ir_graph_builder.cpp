/*
 * Copyright (C) 2019 Samsung Electronics Co. LTD
 *
 * This software is proprietary of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#include "ir/include/ir_graph_builder.hpp"
#include "ir/include/common/log.hpp"
#include "ir/include/ir_blob_builder.hpp"
#include "ir/include/ir_exec_step.hpp"
#include "ir/include/ir_includes.hpp"
#include "ir/include/ir_parser.hpp"
#include "ir/include/ir_tools.hpp"
#include "ir/include/node_execution_step.hpp"

namespace nn_compiler {

/**
 * @brief.      createNodeFromIR
 * @details.    This function creates Node instance from IR
 * @param[in].  node A node of flatbuffer
 * @param[out].
 * @returns.    return code
 */
std::unique_ptr<nn_ir::Node> IRGraphBuilder::createNode(const IR::Node* node, nn_ir::NNIR& graph) {
    IRParser                     ir_parser;
    std::unique_ptr<nn_ir::Node> g_node         = ir_parser.parseNode(node, graph);
    std::string                  operation_mode = "Normal";

    Log::IR::I() << "IRGraphBuilder::createNode() : node id = " << g_node->getId();
    Log::IR::I() << "IRGraphBuilder::createNode() : node name = " << g_node->getName().c_str();
    Log::IR::I() << "IRGraphBuilder::createNode() : original node name = " << g_node->getOriginalNodeName().c_str();
    for (int32_t id : g_node->getInEdgeIds()) {
        Log::IR::I() << "IRGraphBuilder::createNode() : In EdgsIds = " << id;
    }
    for (int32_t id : g_node->getOutEdgeIds()) {
        Log::IR::I() << "IRGraphBuilder::createNode() : Out EdgsIds = " << id;
    }

    graph.setNextNodeId(node->id());

    if (node->hw_info()) {
        auto ir_operation_mode = node->hw_info()->operation_mode();
        if (ir_operation_mode) {
            if (ir_operation_mode->operation_type() != IR::TargetHardware::Type::NodeOperationType_NORMAL) {
                operation_mode = ir_operation_mode->dedicated_operation()->str();
            }
        }
        g_node->setOperationMode(operation_mode);

        if (auto ir_mem_infos = node->hw_info()->mem_info()) {
            std::vector<nn_ir::MemoryInfo> psum_mem_infos;
            std::vector<nn_ir::MemoryInfo> ifm_mem_infos;
            std::vector<nn_ir::MemoryInfo> ofm_mem_infos;
            std::vector<nn_ir::MemoryInfo> kernel_mem_infos;
            std::vector<nn_ir::MemoryInfo> constant_mem_infos;
            std::vector<nn_ir::MemoryInfo> cu_instr_mem_infos;
            std::vector<nn_ir::MemoryInfo> lut_dram_mem_infos;
            std::vector<nn_ir::MemoryInfo> dram_mem_infos;
            for (const auto& ir_mem_info : *ir_mem_infos) {
                const nn_ir::MemoryInfo mem_info = parseMemoryInfo(ir_mem_info);

                if (ir_mem_info->type() == IR::TargetHardware::Type::MemoryType_SRAM ||
                    ir_mem_info->type() == IR::TargetHardware::Type::MemoryType_FIFO) {
                    switch (ir_mem_info->data_type()) {
                        case IR::TargetHardware::Type::MemoryDataType_PSUM:
                            psum_mem_infos.push_back(mem_info);
                            break;
                        case IR::TargetHardware::Type::MemoryDataType_IFM:
                            ifm_mem_infos.push_back(mem_info);
                            break;
                        case IR::TargetHardware::Type::MemoryDataType_OFM:
                            ofm_mem_infos.push_back(mem_info);
                            break;
                        case IR::TargetHardware::Type::MemoryDataType_KERNEL:
                            kernel_mem_infos.push_back(mem_info);
                            break;
                        case IR::TargetHardware::Type::MemoryDataType_INSTR:
                            cu_instr_mem_infos.push_back(mem_info);
                            break;
                        case IR::TargetHardware::Type::MemoryDataType_CONSTANT:
                            constant_mem_infos.push_back(mem_info);
                            break;
                        default:
                            Log::IR::E() << "IRGraphBuilder::createNode(): Unexpected SRAM info type "
                                         << ir_mem_info->data_type() << " on " << *g_node;
                            break;
                    }
                    // FIXME: remove CONSTANT checking condition when PR #5 is merged
                } else if (ir_mem_info->type() == IR::TargetHardware::Type::MemoryType_DRAM &&
                           ir_mem_info->data_type() == IR::TargetHardware::Type::MemoryDataType_CONSTANT) {
                    constant_mem_infos.push_back(mem_info);
                } else if (ir_mem_info->type() == IR::TargetHardware::Type::MemoryType_DRAM &&
                           ir_mem_info->data_type() == IR::TargetHardware::Type::MemoryDataType_LUT) {
                    lut_dram_mem_infos.push_back(mem_info);
                } else {
                    dram_mem_infos.push_back(mem_info);
                }
            }
            g_node->setPsumMemInfo(psum_mem_infos);

            if (g_node->getInEdgeIds().size() == 1) {
                g_node->setIfmMemInfo(g_node->getInEdgeIds().front(), ifm_mem_infos);
            } else {
                auto start_pos = 0;
                auto length    = ifm_mem_infos.size() / g_node->getNumInputs();

                for (auto edge_id : g_node->getInEdgeIds()) {
                    std::vector<nn_ir::MemoryInfo> tmp_ifm_info;
                    for (size_t info_idx = 0; info_idx < length; info_idx++) {
                        tmp_ifm_info.push_back(ifm_mem_infos[start_pos + info_idx]);
                    }
                    g_node->setIfmMemInfo(edge_id, tmp_ifm_info);
                    start_pos += length;
                }
            }
            g_node->setUniqueOfmMemInfo(ofm_mem_infos);

            g_node->setKernelMemInfo(kernel_mem_infos);
            g_node->setConstantMemInfo(constant_mem_infos);

            std::vector<std::pair<nn_ir::MemoryDataType, std::pair<nn_ir::MemoryInfo, nn_ir::MemoryInfo>>>
                cu_instr_mem_info;
            // Since DRAM address is being fetched from DAM instruction, we can just set dummy DRAM info to
            // cu_instr_mem_info.
            nn_ir::MemoryInfo dummy_dram_info = {nn_ir::MemoryType::DRAM, nn_ir::MemoryDataType::INSTR, 0};
            for (auto& sram_mem_info : cu_instr_mem_infos) {
                cu_instr_mem_info.push_back({nn_ir::MemoryDataType::INSTR, {dummy_dram_info, sram_mem_info}});
            }
            g_node->setInstrMemInfo(cu_instr_mem_info);

            // set mem info for LUT blobs
            if (isa<nn_ir::SoftmaxNode>(g_node.get())) {
                auto lut_blobs = nn_ir::getLutBlobs(*g_node);
                lut_blobs.expLUTBlob->setMemoryAllocation(g_node->getId(), {lut_dram_mem_infos[0]});
                lut_blobs.softmaxLUTBlob->setMemoryAllocation(g_node->getId(), {lut_dram_mem_infos[1]});
            }

            Log::IR::E_IF(!dram_mem_infos.empty())
                << "IRGraphBuilder::createNode(): Unexpected DRAM info on " << *g_node;
        }

        typedef const IR::TargetHardware::ExecutionStep* (
            IR::TargetHardware::NodeInfo::*TargetHardware_GetExecutionStep)() const;
        TargetHardware_GetExecutionStep funcList[unsigned(nn_ir::NodeExecutionStepType::COUNT)] = {
            &IR::TargetHardware::NodeInfo::node_data_load_start_step,
            &IR::TargetHardware::NodeInfo::node_data_load_sync_step,
            &IR::TargetHardware::NodeInfo::exec_start_step,
            &IR::TargetHardware::NodeInfo::exec_sync_step};
        for (unsigned type = 0; type < unsigned(nn_ir::NodeExecutionStepType::COUNT); ++type) {
            if (auto step = (node->hw_info()->*(funcList[type]))()) {
                IRExecStep ir_exec_step;
                auto       nstep = ir_exec_step.createNodeExecutionStep(step, graph);
                g_node->setStep(nn_ir::NodeExecutionStepType(type), nstep);
            }
        }
        if (node->hw_info()->mapped_hw()) {
            g_node->setMappedHWName(node->hw_info()->mapped_hw()->str());
        }
    }
    return g_node;
}

/**
 * @brief.      createEdgeFromIR
 * @details.    This function creates Edge instance from IR
 * @param[in].  edge A edge of flatbuffer
 * @param[out].
 * @returns.    return code
 */
std::unique_ptr<nn_ir::Edge> IRGraphBuilder::createEdge(const IR::Edge* edge, nn_ir::NNIR& graph) {
    std::unique_ptr<nn_ir::Edge> g_edge;

    EDGE_ID_T   id          = edge->id();
    std::string name        = edge->name()->str();
    NODE_ID_T   src_node_id = edge->src_node_id();
    NODE_ID_T   dst_node_id = edge->dst_node_id();
    BLOB_ID_T   blob_id     = INVALID_ID;

    graph.setNextEdgeId(id);

    switch (edge->type()) {
        case IR::Type::EdgeType_DATA: {
            blob_id = edge->blob_id();
            g_edge =
                std::make_unique<nn_ir::DataEdge>(nn_ir::EdgeInfo{id, name, graph, src_node_id, dst_node_id}, blob_id);

            Log::IR::I() << "IRGraphBuilder::createEdge() : edge id = " << g_edge->getId();
            Log::IR::I() << "IRGraphBuilder::createEdge() : edge name = " << g_edge->getName().c_str();
            Log::IR::I() << "IRGraphBuilder::createEdge() : blob id = " << blob_id;
            Log::IR::I() << "IRGraphBuilder::createEdge() : in node id = " << g_edge->getInNodeId();
            Log::IR::I() << "IRGraphBuilder::createEdge() : out node id = " << g_edge->getOutNodeId();
            break;
        }
        case IR::Type::EdgeType_CONTROL: {
            Log::IR::E() << "IRGraphBuilder::createEdge() => CONTROL edge is not supported yet!";
            // Need to implement
            break;
        }
        default: {
            Log::IR::E() << "IRGraphBuilder::createEdge() => unknown edge type!";
        }
    }

    if (edge->hw_info()) {
        typedef const IR::TargetHardware::ExecutionStep* (IR::TargetHardware::EdgeInfo::*TargetHardware_GetExecStep)()
            const;
        TargetHardware_GetExecStep funcList[unsigned(nn_ir::EdgeExecutionStepType::COUNT)] = {
            &IR::TargetHardware::EdgeInfo::load_start_step,
            &IR::TargetHardware::EdgeInfo::load_sync_step,
            &IR::TargetHardware::EdgeInfo::store_start_step,
            &IR::TargetHardware::EdgeInfo::store_sync_step};
        for (unsigned type = 0; type < unsigned(nn_ir::EdgeExecutionStepType::COUNT); ++type) {
            if (auto step = (edge->hw_info()->*funcList[type])()) {
                IRExecStep ir_exec_step;
                auto       nstep = ir_exec_step.createEdgeExecutionStep(step, graph);
                g_edge->setStep(nn_ir::EdgeExecutionStepType(type), nstep);
            }
        }

        if (auto ir_mem_infos = edge->hw_info()->mem_info()) {
            nn_ir::Blob* blob = graph.getBlob(blob_id);
            Log::IR::E_IF(!blob) << "Edge #" << id << ": Blob #" << blob_id << " has not been loaded";
            std::vector<nn_ir::MemoryInfo> mem_infos;

            for (const IR::TargetHardware::Type::MemoryInfo* ir_mem_info : *ir_mem_infos) {
                mem_infos.push_back(parseMemoryInfo(ir_mem_info));
            }

            blob->setMemoryAllocation(id, mem_infos);
        }
    }
    return g_edge;
}
} // namespace nn_compiler
