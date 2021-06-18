/*
 * Copyright (C) 2019 Samsung Electronics Co. LTD
 *
 * This software is proprietary of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

/**
 * @file.    ir_exporter.hpp
 * @brief.   This is IRExporter class
 * @details. This header defines IRExporter class.
 * @version. 0.1.
 */

#pragma once

#include "flatbuffers/flatbuffers.h"

#include "common/common.hpp"
#include "common/types.hpp"

#include "ir/generated/ir_generated.h"
#include "ir/nn_ir.hpp"

#include "ir/blob.hpp"
#include "ir/edge.hpp"

#include "ir/ir_types.hpp"

#include "ir/nn_node.hpp"
#include "ir/op_node.hpp"
#include "ir/q_node.hpp"

namespace nn_compiler {

class IRExporter {
 public:
    /**
     * @brief.      Write a file for nn_ir::NNIR
     * @details.    This function stores nn_ir::NNIR class to the file
     * @param[in].  file_path Output file path
     * @param[in].  graphs A list of nn_ir::NNIR to be written
     * @param[out].
     * @returns.    return code
     */
    RetVal generateFileFromNNIR(const std::string& file_path, const std::vector<std::unique_ptr<nn_ir::NNIR>>& graphs);

    /**
     * @brief.      To get IRExporter instance (Singleton)
     * @returns.    A pointer of IRExporter instance
     */
    static IRExporter* getInstance() {
        static IRExporter instance_;
        return &instance_;
    }

 private:
    /**
     * @brief.      Constructor of IRExporter.
     * @details.    This function constructs IRExporter
     * @param[in].
     * @param[out].
     * @returns.
     */
    IRExporter();

    /**
     * @brief.      save IR from blob
     * @details.    This function creates IR::Blob instance from nn_ir::Blob
     * @returns.    flatbuffers::Offset<IR::Node>
     */
    flatbuffers::Offset<IR::Blob> saveBlob(flatbuffers::FlatBufferBuilder& builder, const nn_ir::Blob& nn_blob);

    /**
     * @brief.      save IR from node
     * @details.    This function creates IR::Node instance from nn_ir::Node
     * @returns.    flatbuffers::Offset<IR::Node>
     */
    flatbuffers::Offset<IR::Node> saveNode(flatbuffers::FlatBufferBuilder& builder, const nn_ir::Node& node);

    /**
     * @brief.      save IR from Edge
     * @details.    This function creates IR::Edge instance from nn_ir::Edge
     * @returns.    flatbuffers::Offset<IR::Edge>
     */
    flatbuffers::Offset<IR::Edge> saveEdge(flatbuffers::FlatBufferBuilder& builder, const nn_ir::Edge& nn_edge);

    /**
     * @brief.      save IR from ExecutionStep
     * @details.    This function creates IR::TargetHardware::ExecutionStep instance from nn_ir::ExecutionStep
     * @returns.    flatbuffers::Offset<IR::TargetHardware::ExecutionStep>
     */
    flatbuffers::Offset<IR::TargetHardware::ExecutionStep> saveStep(flatbuffers::FlatBufferBuilder& builder,
                                                                    const nn_ir::ExecutionStep&     nn_step);

    /**
     * @brief.      save IR from Instruction
     * @details.    This function creates IR::TargetHardware::Instruction instance from nn_ir::Instruction
     * @returns.    flatbuffers::Offset<IR::TargetHardware::Instruction>
     */
    flatbuffers::Offset<IR::TargetHardware::Instruction> saveInstr(flatbuffers::FlatBufferBuilder& builder,
                                                                   const nn_ir::Instruction&       nn_instr);

    template <nn_ir::NodeType>
    flatbuffers::Offset<IR::HwNode> makeHWNodeIR(flatbuffers::FlatBufferBuilder& builder, const nn_ir::HWNode& nn_node);

    template <nn_ir::NodeType>
    flatbuffers::Offset<IR::NnNode> makeNNNodeIR(flatbuffers::FlatBufferBuilder& builder, const nn_ir::NNNode& nn_node);

    template <nn_ir::NodeType>
    flatbuffers::Offset<IR::OpNode> makeOPNodeIR(flatbuffers::FlatBufferBuilder& builder, const nn_ir::OPNode& op_node);

    template <nn_ir::NodeType>
    flatbuffers::Offset<IR::globalNode> makeGlobalNodeIR(flatbuffers::FlatBufferBuilder& builder,
                                                         const nn_ir::GlobalNode&        g_node);
    template <nn_ir::NodeType>
    flatbuffers::Offset<IR::vNode> makeVNodeIR(flatbuffers::FlatBufferBuilder& builder, const nn_ir::VNode& v_node);

    template <nn_ir::NodeType>
    flatbuffers::Offset<IR::qNode> makeQNodeIR(flatbuffers::FlatBufferBuilder& builder, const nn_ir::QNode& q_node);
    flatbuffers::Offset<IR::NNNode::ActivationNode> makeActivation(flatbuffers::FlatBufferBuilder& builder,
                                                                   const nn_ir::ActivationNode&    act_node);
    flatbuffers::Offset<IR::OPNode::ShiftNode>      makeShift(flatbuffers::FlatBufferBuilder& builder,
                                                              const nn_ir::ShiftNode*         shift_node);

    using nnMakeFunction = flatbuffers::Offset<IR::NnNode> (IRExporter::*)(flatbuffers::FlatBufferBuilder& builder,
                                                                           const nn_ir::NNNode&            nn_node);
    using hwMakeFunction = flatbuffers::Offset<IR::HwNode> (IRExporter::*)(flatbuffers::FlatBufferBuilder& builder,
                                                                           const nn_ir::HWNode&            hw_node);
    using opMakeFunction = flatbuffers::Offset<IR::OpNode> (IRExporter::*)(flatbuffers::FlatBufferBuilder& builder,
                                                                           const nn_ir::OPNode&            nn_node);
    using gMakeFunction  = flatbuffers::Offset<IR::globalNode> (IRExporter::*)(flatbuffers::FlatBufferBuilder& builder,
                                                                              const nn_ir::GlobalNode&        nn_node);
    using vMakeFunction  = flatbuffers::Offset<IR::vNode> (IRExporter::*)(flatbuffers::FlatBufferBuilder& builder,
                                                                         const nn_ir::VNode&             nn_node);
    using qMakeFunction  = flatbuffers::Offset<IR::qNode> (IRExporter::*)(flatbuffers::FlatBufferBuilder& builder,
                                                                         const nn_ir::QNode&             q_node);

    std::map<nn_ir::NodeType, nnMakeFunction> nn_node_ir_make_func_map_;
    std::map<nn_ir::NodeType, hwMakeFunction> hw_node_ir_make_func_map_;
    std::map<nn_ir::NodeType, opMakeFunction> op_node_ir_make_func_map_;
    std::map<nn_ir::NodeType, gMakeFunction>  g_node_ir_make_func_map_;
    std::map<nn_ir::NodeType, vMakeFunction>  v_node_ir_make_func_map_;
    std::map<nn_ir::NodeType, qMakeFunction>  q_node_ir_make_func_map_;
}; // class IRExporter

} // namespace nn_compiler
