/*
 * Copyright (C) 2019 Samsung Electronics Co. LTD
 *
 * This software is proprietary of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#include "ir/include/ir_exporter.hpp"
#include "ir/include/common/log.hpp"
#include "ir/include/ir_includes.hpp"
#include "ir/include/ir_types.hpp"
#include "ir/include/op_nodes/shift_node.hpp"

namespace nn_compiler {

/**
 * @brief.      Write a file for nn_ir::NNIR
 * @details.    This function stores nn_ir::NNIR class to the file
 * @param[in].  file_path Output file path
 * @param[in].  graphs A list of nn_ir::NNIR to be written
 * @param[out].
 * @returns.    return code
 */
RetVal IRExporter::generateFileFromNNIR(const std::string&                               file_path,
                                        const std::vector<std::unique_ptr<nn_ir::NNIR>>& graphs) {
    Log::IR::I() << "IRExporter::generateFileFromNNIR() is called, path: " << file_path;

    flatbuffers::FlatBufferBuilder              builder;
    std::vector<flatbuffers::Offset<IR::Graph>> ir_graph;
    for (const auto& graph : graphs) {
        std::vector<flatbuffers::Offset<IR::Node>> ir_node;
        for (const auto& node : graph->getNodes()) {
            ir_node.emplace_back(saveNode(builder, node));
        }

        std::vector<flatbuffers::Offset<IR::Edge>> ir_edge;
        for (const auto& edge : graph->getEdges()) {
            ir_edge.emplace_back(saveEdge(builder, edge));
        }

        std::vector<flatbuffers::Offset<IR::Blob>> ir_blob;
        for (const auto& blob : graph->getBlobs()) {
            ir_blob.emplace_back(saveBlob(builder, blob));
        }

        auto graph_name = builder.CreateString(graph->getName());

        // TODO(s-steve-jang): NEED TO DEFINE NODEGROUP (BELOW IS A DUMMY)
        std::vector<flatbuffers::Offset<IR::NodeGroup>> node_group;
        node_group.push_back(IR::CreateNodeGroup(builder));
        flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<IR::NodeGroup>>> ir_node_group;
        ir_node_group = builder.CreateVector(node_group);

        flatbuffers::Offset<IR::TargetHardware::GraphInfo> hw_graph_info;
        hw_graph_info = IR::TargetHardware::CreateGraphInfo(builder, builder.CreateVector(graph->getExecutionSteps()));
        ir_graph.emplace_back(IR::CreateGraph(builder,
                                              graph->getId(),
                                              graph_name,
                                              ir_node_group,
                                              builder.CreateVector(ir_node),
                                              builder.CreateVector(ir_edge),
                                              builder.CreateVector(ir_blob),
                                              hw_graph_info));
    }
    auto root = IR::CreateRoot(builder, builder.CreateVector(ir_graph));
    builder.Finish(root);

    std::ofstream outfile;
    uint8_t*      buffer = builder.GetBufferPointer();
    int           size   = builder.GetSize();
    outfile.exceptions(std::ofstream::failbit | std::ofstream::badbit);
    // Open Output File
    try {
        outfile.open(file_path.c_str(), std::ios::binary | std::ios::out);
        outfile.write(reinterpret_cast<char*>(buffer), size);
        outfile.close();
    }
    catch (const std::ifstream::failure& e) {
        Log::IR::E() << "IRExporter::generateFileFromNNIR() : FileSaveFail, path: " << file_path;
        return RetVal::FAILURE;
    }
    return RetVal::SUCCESS;
}

static IR::Type::Dim4 saveDim4(const nn_ir::Shape4D& dim) { return IR::Type::Dim4(dim.n, dim.c, dim.h, dim.w); }
static IR::Type::Dim4 saveCoordinate(const nn_ir::Coord4D& coord) {
    return IR::Type::Dim4(coord.n, coord.c, coord.h, coord.w);
}

/**
 * @brief.      save IR from blob
 * @details.    This function creates IR::Blob instance from nn_ir::Blob
 * @param[out].
 * @returns.    flatbuffers::Offset<IR::Bolb>
 */
flatbuffers::Offset<IR::Blob> IRExporter::saveBlob(flatbuffers::FlatBufferBuilder& builder,
                                                   const nn_ir::Blob&              nn_blob) {
    auto id             = nn_blob.getId();
    auto ir_blob_name   = builder.CreateString(nn_blob.getName());
    auto graph_id       = nn_blob.getGraph().getId();
    auto bit_width      = nn_blob.getBitWidth();
    auto dim            = nn_blob.getShape();
    auto alignment_unit = nn_blob.getSizeAlignment();
    auto data_type      = nn_blob.getDataType();
    auto liveness       = nn_blob.getLiveness();
    auto zero_point     = nn_blob.getZeroPoint();
    auto compress       = nn_blob.getCompress() ? IR::TargetHardware::Type::CompressionType_FLC
                                          : IR::TargetHardware::Type::CompressionType_NONE;

    auto frac_len = nn_blob.getFracLen();
    // FIXME: ADDITIONAL ATTRIBUTES SHOULD BE INCLUDED
    // auto quant_level    = nn_blob.getQuantLevel();
    // auto ir_quant_level     = IR::Type::QuantLevelType(quant_level);
    auto ir_dim            = saveDim4(dim);
    auto ir_alignment_unit = saveDim4(alignment_unit);

    flatbuffers::Offset<IR::TargetHardware::Type::CompressionInfo> hw_compression_info;
    flatbuffers::Offset<IR::TargetHardware::BlobInfo>              hw_blob_info;
    flatbuffers::Offset<flatbuffers::Vector<int8_t>>               ir_frac_len = builder.CreateVector(frac_len);

    nn_ir::NNIR_Node_Config_Type_ nnir_blob_type = nn_blob.getBlobType();
    IR::Type::BlobType            ir_blob_type   = std::get<IR::Type::BlobType>(nn_ir::parseConfigType(nnir_blob_type));

    nn_ir::NNIR_Node_Config_Type_ nnir_quant_type = nn_blob.getQuantType();
    IR::Type::QuantType ir_quant_type = std::get<IR::Type::QuantType>(nn_ir::parseConfigType(nnir_quant_type));

    nn_ir::NNIR_Node_Config_Type_ nnir_shape_type = nn_blob.getShapeType();
    IR::Type::ShapeType ir_shape_type = std::get<IR::Type::ShapeType>(nn_ir::parseConfigType(nnir_shape_type));

#define DATATYPE(NNIR_TYPE, IR_GENERATED_TYPE, C_TYPE)                                                                \
    case nn_ir::DataType::NNIR_TYPE: {                                                                                \
        ir_data_type = IR::Type::DataType_##IR_GENERATED_TYPE;                                                        \
        if (auto arr_data_blob = cast_if<nn_ir::DataBlob>(nn_blob)) {                                                 \
            const auto& original_buf = arr_data_blob->getBuf<C_TYPE>();                                               \
            if (!original_buf.empty()) {                                                                              \
                auto original_buf_ptr = reinterpret_cast<const uint8_t*>(original_buf.data());                        \
                ir_data_arr           = builder.CreateVector(original_buf_ptr, original_buf.size() * sizeof(C_TYPE)); \
            }                                                                                                         \
        }                                                                                                             \
        break;                                                                                                        \
    }

    IR::Type::DataType                                ir_data_type;
    flatbuffers::Offset<flatbuffers::Vector<uint8_t>> ir_data_arr;
    switch (data_type) {
        DATATYPE(FLOAT32, FP_32, float)
        DATATYPE(INT32, FIXED_32, int32_t)
        DATATYPE(FLOAT16, FP_16, float16)
        DATATYPE(INT16, FIXED_16, int16_t)
        DATATYPE(UINT16, FIXED_16U, uint16_t)
        DATATYPE(INT8, FIXED_8, int8_t)
        DATATYPE(UINT8, FIXED_8U, uint8_t)
        DATATYPE(INT64, FIXED_64, int64_t)
        DATATYPE(INT4, FIXED_4, int4_t)
        DATATYPE(UINT4, FIXED_4U, uint4_t)
        default:
            Log::IR::E() << "IRExporter::saveBlob() => unknown data type!";
            break;
    }
#undef DATATYPE

    // Compression info
    uint32_t                                          compressed_data_size = 0;
    uint32_t                                          metadata_size        = 0;
    flatbuffers::Offset<flatbuffers::Vector<uint8_t>> ir_metadata_arr      = 0;
    switch (nn_blob.getBlobType()) {
        case nn_ir::BlobType::WEIGHT:

#define DATATYPE(NNIR_TYPE, C_TYPE)                                                                          \
    case nn_ir::DataType::NNIR_TYPE: {                                                                       \
        if (auto data_blob = cast_if<nn_ir::DataBlob>(nn_blob)) {                                            \
            auto meta_buf = data_blob->getMetaBuf();                                                         \
            metadata_size = meta_buf.size();                                                                 \
            if (!meta_buf.empty()) {                                                                         \
                compress             = IR::TargetHardware::Type::CompressionType_FLC;                        \
                compressed_data_size = data_blob->getBuf<C_TYPE>().size();                                   \
                auto metabuf_ptr     = reinterpret_cast<const uint8_t*>(meta_buf.data());                    \
                ir_metadata_arr      = builder.CreateVector(metabuf_ptr, meta_buf.size() * sizeof(uint8_t)); \
            }                                                                                                \
        }                                                                                                    \
        break;                                                                                               \
    }

            switch (data_type) {
                DATATYPE(FLOAT32, float)
                DATATYPE(FLOAT16, float16)
                DATATYPE(INT16, int16_t)
                DATATYPE(UINT16, uint16_t)
                DATATYPE(INT8, int8_t)
                DATATYPE(UINT8, uint8_t)
                DATATYPE(INT64, int64_t)
                default:
                    Log::IR::E() << "IRExporter::saveNode() => unknown node type!";
                    break;
            }
#undef DATATYPE
            break;
        case nn_ir::BlobType::FEATUREMAP:
        case nn_ir::BlobType::BIAS:
        case nn_ir::BlobType::LUT:
            break;
        default:
            Log::IR::E() << "IRExporter::saveNode() => unknown node type!";
            break;
    }

    hw_compression_info = IR::TargetHardware::Type::CreateCompressionInfo(
        builder, compress, compressed_data_size, metadata_size, ir_metadata_arr, nn_blob.getFLCFragments());
    hw_blob_info = IR::TargetHardware::CreateBlobInfo(
        builder, liveness.first, liveness.second, &ir_alignment_unit, hw_compression_info);
    return IR::CreateBlob(builder,
                          id,
                          ir_blob_name,
                          graph_id,
                          ir_blob_type,
                          ir_quant_type,
                          ir_shape_type,
                          &ir_dim,
                          ir_data_type,
                          ir_data_arr,
                          bit_width,
                          zero_point,
                          IR::Type::QuantLevelType_LAYERWISE, // dummy
                          ir_frac_len,
                          hw_blob_info);
}

static flatbuffers::Offset<IR::TargetHardware::Type::DataLayout> saveDataLayout(flatbuffers::FlatBufferBuilder& builder,
                                                                                const nn_ir::DataLayout& layout) {
    IR::Type::Dim4 total_dim = saveDim4(layout.total_dim);
    IR::Type::Pad4 padding(layout.padding.l, layout.padding.r, layout.padding.t, layout.padding.b);
    IR::Type::Dim4 gap       = saveDim4(layout.gap);
    IR::Type::Dim4 cell_unit = saveDim4(layout.cell_unit);

    nn_ir::NNIR_Node_Config_Type_ byte_order = layout.byte_order;
    auto cell_byte_order = std::get<IR::TargetHardware::Type::PixelByteOrder>(nn_ir::parseConfigType(byte_order));

    flatbuffers::Offset<IR::TargetHardware::Type::CellInfo> cellinfo =
        IR::TargetHardware::Type::CreateCellInfo(builder, &cell_unit, layout.bpp, cell_byte_order);

    return IR::TargetHardware::Type::CreateDataLayout(builder, &total_dim, &padding, &gap, cellinfo);
}

static flatbuffers::Offset<IR::TargetHardware::Type::MemoryInfo> saveMemoryInfo(flatbuffers::FlatBufferBuilder& builder,
                                                                                const nn_ir::MemoryInfo& mem_info) {
    nn_ir::NNIR_Node_Config_Type_        memory_type   = mem_info.memory_type;
    nn_ir::NNIR_Node_Config_Type_        memory_region = mem_info.data_type;
    IR::TargetHardware::Type::MemoryType ir_mem_type =
        std::get<IR::TargetHardware::Type::MemoryType>(nn_ir::parseConfigType(memory_type));
    IR::TargetHardware::Type::MemoryDataType ir_mem_region =
        std::get<IR::TargetHardware::Type::MemoryDataType>(nn_ir::parseConfigType(memory_region));
    flatbuffers::Offset<IR::TargetHardware::Type::DataLayout> layout = saveDataLayout(builder, mem_info.layout);

    return IR::TargetHardware::Type::CreateMemoryInfo(
        builder, ir_mem_type, ir_mem_region, mem_info.mem_id, mem_info.size, mem_info.addr, layout);
}

/**
 * @brief.      save IR from node
 * @details.    This function creates IR::Node instance from nn_ir::Node
 */
flatbuffers::Offset<IR::Node> IRExporter::saveNode(flatbuffers::FlatBufferBuilder& builder, const nn_ir::Node& node) {
    flatbuffers::Offset<IR::Node> ir_node;
    auto                          id           = node.getId();
    auto                          ir_node_name = builder.CreateString(node.getName());
    auto                          graph_id     = node.getGraph().getId();
    auto                          in_edge_ids  = node.getInEdgeIds();
    auto                          out_edge_ids = node.getOutEdgeIds();

    auto                                     operation_type = IR::TargetHardware::Type::NodeOperationType_NORMAL;
    flatbuffers::Offset<flatbuffers::String> ir_dedicated_operation = 0;
    if (!node.getOperationMode().empty() && node.getOperationMode() != "Normal") {
        operation_type         = IR::TargetHardware::Type::NodeOperationType_DEDICATED;
        ir_dedicated_operation = builder.CreateString(node.getOperationMode());
    }
    auto ir_operation_mode = IR::TargetHardware::CreateOperationMode(builder, operation_type, ir_dedicated_operation);

    auto ir_mapped_hw = builder.CreateString(node.getMappedHWName());

    flatbuffers::Offset<flatbuffers::Vector<EDGE_ID_T>> ir_in_edge_ids;
    if (in_edge_ids.size() > 0) {
        ir_in_edge_ids = builder.CreateVector(in_edge_ids);
    }
    flatbuffers::Offset<flatbuffers::Vector<EDGE_ID_T>> ir_out_edge_ids;
    if (out_edge_ids.size() > 0) {
        ir_out_edge_ids = builder.CreateVector(out_edge_ids);
    }

    flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<IR::TargetHardware::Type::MemoryInfo>>> ir_mem_infos;
    std::vector<flatbuffers::Offset<IR::TargetHardware::Type::MemoryInfo>>                              mem_infos;

    auto save_mem_infos = [&builder, &mem_infos](const std::vector<nn_ir::MemoryInfo>&    data_info,
                                                 IR::TargetHardware::Type::MemoryDataType memory_data_type) {
        for (const auto& mem_info : data_info) {
            Log::IR::E_IF(mem_info.addr == INVALID_OFFSET || mem_info.data_type == nn_ir::MemoryDataType::INVALID)
                << "Trying to export invalid meminfo: " << mem_info;
            flatbuffers::Offset<IR::TargetHardware::Type::DataLayout> layout = saveDataLayout(builder, mem_info.layout);
            auto memory_type = mem_info.memory_type == nn_ir::MemoryType::FIFO
                                   ? IR::TargetHardware::Type::MemoryType_FIFO
                                   : mem_info.memory_type == nn_ir::MemoryType::SRAM
                                         ? IR::TargetHardware::Type::MemoryType_SRAM
                                         : IR::TargetHardware::Type::MemoryType_DRAM;
            mem_infos.push_back(IR::TargetHardware::Type::CreateMemoryInfo(
                builder, memory_type, memory_data_type, mem_info.mem_id, mem_info.size, mem_info.addr, layout));
        }
    };

    // save psum
    save_mem_infos(node.getPsumMemInfo(), IR::TargetHardware::Type::MemoryDataType_PSUM);

    // save ifm and ofm
    for (uint32_t i = 0; i < node.getNumInputs(); ++i) {
        save_mem_infos(node.getIfmMemInfoByInputNo(i), IR::TargetHardware::Type::MemoryDataType_IFM);
    }
    for (uint32_t i = 0; i < node.getNumOutputs(); ++i) {
        save_mem_infos(node.getOfmMemInfoByOutputNo(i), IR::TargetHardware::Type::MemoryDataType_OFM);
    }

    // save kernel
    save_mem_infos(node.getKernelMemInfo(), IR::TargetHardware::Type::MemoryDataType_KERNEL);

    // save constant
    save_mem_infos(node.getConstantMemInfo(), IR::TargetHardware::Type::MemoryDataType_CONSTANT);

    // save cu_instr
    std::vector<nn_ir::MemoryInfo> cu_instr_sram_mem_infos;
    const auto&                    instr_memory_info = node.getInstrMemInfos();
    for (const auto& mem_info : instr_memory_info) {
        cu_instr_sram_mem_infos.push_back(mem_info.second.second);
    }
    save_mem_infos(cu_instr_sram_mem_infos, IR::TargetHardware::Type::MemoryDataType_INSTR);

    // save Softmax LUT blobs DRAM
    if (isa<nn_ir::SoftmaxNode>(node)) {
        std::vector<nn_ir::MemoryInfo> lut_dram_mem_infos;
        auto                           lut_blobs = nn_ir::getLutBlobs(node);
        lut_dram_mem_infos.push_back(lut_blobs.expLUTBlob->getFirstMemoryAllocation(node.getId()));
        lut_dram_mem_infos.push_back(lut_blobs.softmaxLUTBlob->getFirstMemoryAllocation(node.getId()));
        save_mem_infos(lut_dram_mem_infos, IR::TargetHardware::Type::MemoryDataType_LUT);
    }

    ir_mem_infos = builder.CreateVector(mem_infos);

    // save steps
    std::vector<flatbuffers::Offset<IR::TargetHardware::ExecutionStep>> ir_steps;
    for (unsigned type = 0; type < (unsigned)nn_ir::NodeExecutionStepType::COUNT; ++type) {
        ir_steps.push_back(saveStep(builder, node.getStep((nn_ir::NodeExecutionStepType)type)));
    }

    if (isa<nn_ir::NNNode>(node)) {
        const auto& nn_node = static_cast<const nn_ir::NNNode&>(node);
        const auto& it      = nn_node_ir_make_func_map_.find(nn_node.getNodeType());
        Log::IR::E_IF(it == nn_node_ir_make_func_map_.end()) << "IRExporter::saveNode() => unknown nn node type!";

        const auto&                     maker     = it->second;
        flatbuffers::Offset<IR::NnNode> ir_NNNode = (this->*maker)(builder, nn_node);

        flatbuffers::Offset<IR::TargetHardware::NodeInfo> hw_node_info;
        hw_node_info =
            IR::TargetHardware::CreateNodeInfo(builder,
                                               ir_mem_infos,
                                               ir_steps[(unsigned)nn_ir::NodeExecutionStepType::NODE_DATA_LOAD_START],
                                               ir_steps[(unsigned)nn_ir::NodeExecutionStepType::NODE_DATA_LOAD_SYNC],
                                               ir_steps[(unsigned)nn_ir::NodeExecutionStepType::EXEC_START],
                                               ir_steps[(unsigned)nn_ir::NodeExecutionStepType::EXEC_SYNC],
                                               ir_mapped_hw,
                                               ir_operation_mode);
        ir_node = IR::CreateNode(builder,
                                 id,
                                 ir_node_name,
                                 graph_id,
                                 IR::AnyNode_NnNode,
                                 ir_NNNode.Union(),
                                 ir_in_edge_ids,
                                 ir_out_edge_ids,
                                 hw_node_info);
    } else if (isa<nn_ir::OPNode>(node)) {
        const auto& op_node = static_cast<const nn_ir::OPNode&>(node);
        const auto& it      = op_node_ir_make_func_map_.find(op_node.getNodeType());
        Log::IR::E_IF(it == op_node_ir_make_func_map_.end()) << "IRExporter::saveNode() => unknown op node type!";
        const auto&                                       maker     = it->second;
        flatbuffers::Offset<IR::OpNode>                   ir_OPNode = (this->*maker)(builder, op_node);
        flatbuffers::Offset<IR::TargetHardware::NodeInfo> hw_node_info =
            IR::TargetHardware::CreateNodeInfo(builder,
                                               ir_mem_infos,
                                               ir_steps[(unsigned)nn_ir::NodeExecutionStepType::NODE_DATA_LOAD_START],
                                               ir_steps[(unsigned)nn_ir::NodeExecutionStepType::NODE_DATA_LOAD_SYNC],
                                               ir_steps[(unsigned)nn_ir::NodeExecutionStepType::EXEC_START],
                                               ir_steps[(unsigned)nn_ir::NodeExecutionStepType::EXEC_SYNC],
                                               ir_mapped_hw,
                                               ir_operation_mode);
        ir_node = IR::CreateNode(builder,
                                 id,
                                 ir_node_name,
                                 graph_id,
                                 IR::AnyNode_OpNode,
                                 ir_OPNode.Union(),
                                 ir_in_edge_ids,
                                 ir_out_edge_ids,
                                 hw_node_info);
    } else if (isa<nn_ir::VNode>(node)) {
        const auto& v_node = static_cast<const nn_ir::VNode&>(node);
        const auto& it     = v_node_ir_make_func_map_.find(v_node.getNodeType());
        Log::IR::E_IF(it == v_node_ir_make_func_map_.end()) << "IRExporter::saveNode() => unknown v node type!";

        const auto&                                       maker    = it->second;
        flatbuffers::Offset<IR::vNode>                    ir_VNode = (this->*maker)(builder, v_node);
        flatbuffers::Offset<IR::TargetHardware::NodeInfo> hw_node_info =
            IR::TargetHardware::CreateNodeInfo(builder,
                                               ir_mem_infos,
                                               ir_steps[(unsigned)nn_ir::NodeExecutionStepType::NODE_DATA_LOAD_START],
                                               ir_steps[(unsigned)nn_ir::NodeExecutionStepType::NODE_DATA_LOAD_SYNC],
                                               ir_steps[(unsigned)nn_ir::NodeExecutionStepType::EXEC_START],
                                               ir_steps[(unsigned)nn_ir::NodeExecutionStepType::EXEC_SYNC]);
        ir_node = IR::CreateNode(builder,
                                 id,
                                 ir_node_name,
                                 graph_id,
                                 IR::AnyNode_vNode,
                                 ir_VNode.Union(),
                                 ir_in_edge_ids,
                                 ir_out_edge_ids,
                                 hw_node_info);
    } else if (isa<nn_ir::GlobalNode>(node)) {
        const auto& g_node = static_cast<const nn_ir::GlobalNode&>(node);
        const auto& it     = g_node_ir_make_func_map_.find(g_node.getNodeType());
        Log::IR::E_IF(it == g_node_ir_make_func_map_.end()) << "IRExporter::saveNode() => unknown g node type!";

        const auto&                                       maker         = it->second;
        flatbuffers::Offset<IR::globalNode>               ir_GlobalNode = (this->*maker)(builder, g_node);
        flatbuffers::Offset<IR::TargetHardware::NodeInfo> hw_node_info =
            IR::TargetHardware::CreateNodeInfo(builder,
                                               ir_mem_infos,
                                               ir_steps[(unsigned)nn_ir::NodeExecutionStepType::NODE_DATA_LOAD_START],
                                               ir_steps[(unsigned)nn_ir::NodeExecutionStepType::NODE_DATA_LOAD_SYNC],
                                               ir_steps[(unsigned)nn_ir::NodeExecutionStepType::EXEC_START],
                                               ir_steps[(unsigned)nn_ir::NodeExecutionStepType::EXEC_SYNC]);
        ir_node = IR::CreateNode(builder,
                                 id,
                                 ir_node_name,
                                 graph_id,
                                 IR::AnyNode_globalNode,
                                 ir_GlobalNode.Union(),
                                 ir_in_edge_ids,
                                 ir_out_edge_ids,
                                 hw_node_info);
    } else if (isa<nn_ir::QNode>(node)) {
        const auto& q_node = static_cast<const nn_ir::QNode&>(node);
        const auto& it     = q_node_ir_make_func_map_.find(q_node.getNodeType());
        Log::IR::E_IF(it == q_node_ir_make_func_map_.end()) << "IRExporter::saveNode() => unknown q node type!";

        const auto&                                       maker    = it->second;
        flatbuffers::Offset<IR::qNode>                    ir_QNode = (this->*maker)(builder, q_node);
        flatbuffers::Offset<IR::TargetHardware::NodeInfo> hw_node_info =
            IR::TargetHardware::CreateNodeInfo(builder,
                                               ir_mem_infos,
                                               ir_steps[(unsigned)nn_ir::NodeExecutionStepType::NODE_DATA_LOAD_START],
                                               ir_steps[(unsigned)nn_ir::NodeExecutionStepType::NODE_DATA_LOAD_SYNC],
                                               ir_steps[(unsigned)nn_ir::NodeExecutionStepType::EXEC_START],
                                               ir_steps[(unsigned)nn_ir::NodeExecutionStepType::EXEC_SYNC],
                                               ir_mapped_hw,
                                               ir_operation_mode);
        ir_node = IR::CreateNode(builder,
                                 id,
                                 ir_node_name,
                                 graph_id,
                                 IR::AnyNode_qNode,
                                 ir_QNode.Union(),
                                 ir_in_edge_ids,
                                 ir_out_edge_ids,
                                 hw_node_info);
    } else if (isa<nn_ir::HWNode>(node)) {
        flatbuffers::Offset<IR::HwNode> ir_HWNode;
        const auto&                     hw_node = static_cast<const nn_ir::HWNode&>(node);
        const auto&                     it      = hw_node_ir_make_func_map_.find(hw_node.getNodeType());
        if (it == hw_node_ir_make_func_map_.end()) {
            Log::IR::E() << "IRExporter::saveNode() => unknown node type!";
        } else {
            const auto& maker = it->second;
            ir_HWNode         = (this->*maker)(builder, hw_node);
        }
        flatbuffers::Offset<IR::TargetHardware::NodeInfo> hw_node_info;
        hw_node_info =
            IR::TargetHardware::CreateNodeInfo(builder,
                                               ir_mem_infos,
                                               ir_steps[(unsigned)nn_ir::NodeExecutionStepType::NODE_DATA_LOAD_START],
                                               ir_steps[(unsigned)nn_ir::NodeExecutionStepType::NODE_DATA_LOAD_SYNC],
                                               ir_steps[(unsigned)nn_ir::NodeExecutionStepType::EXEC_START],
                                               ir_steps[(unsigned)nn_ir::NodeExecutionStepType::EXEC_SYNC],
                                               ir_mapped_hw,
                                               ir_operation_mode);
        ir_node = IR::CreateNode(builder,
                                 id,
                                 ir_node_name,
                                 graph_id,
                                 IR::AnyNode_HwNode,
                                 ir_HWNode.Union(),
                                 ir_in_edge_ids,
                                 ir_out_edge_ids,
                                 hw_node_info);
    } else {
        Log::IR::E() << "IRExporter::saveNode() => unknown node type!";
    }

    return ir_node;
}

/**
 * @brief.      save IR from Edge
 * @details.    This function creates IR::Edge instance from nn_ir::Edge
 */
flatbuffers::Offset<IR::Edge> IRExporter::saveEdge(flatbuffers::FlatBufferBuilder& builder,
                                                   const nn_ir::Edge&              nn_edge) {
    auto ir_edge_id     = nn_edge.getId();
    auto ir_edge_name   = builder.CreateString(nn_edge.getName());
    auto ir_graph_id    = nn_edge.getGraph().getId();
    auto ir_in_node_id  = nn_edge.getInNodeId();
    auto ir_out_node_id = nn_edge.getOutNodeId();
    auto edge_type      = nn_edge.getEdgeType();

    // save steps
    std::vector<flatbuffers::Offset<IR::TargetHardware::ExecutionStep>> ir_steps;
    for (unsigned type = 0; type < (unsigned)nn_ir::EdgeExecutionStepType::COUNT; ++type) {
        ir_steps.push_back(saveStep(builder, nn_edge.getStep((nn_ir::EdgeExecutionStepType)type)));
    }

    switch (edge_type) {
        case nn_ir::EdgeType::DATA: {
            auto&              data_edge  = cast<nn_ir::DataEdge>(nn_edge);
            BLOB_ID_T          ir_blob_id = data_edge.getBlobId();
            IR::Type::EdgeType ir_edge_type(IR::Type::EdgeType_DATA);
            const nn_ir::Blob* blob = data_edge.getBlob();

            flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<IR::TargetHardware::Type::MemoryInfo>>>
                ir_mem_infos;

            if (blob) {
                std::vector<flatbuffers::Offset<IR::TargetHardware::Type::MemoryInfo>> mem_infos;

                for (auto& mem_info : blob->getMemoryAllocation(ir_edge_id)) {
                    mem_infos.push_back(saveMemoryInfo(builder, mem_info));
                }

                ir_mem_infos = builder.CreateVector(mem_infos);
            }

            flatbuffers::Offset<IR::TargetHardware::EdgeInfo> hw_edge_info =
                IR::TargetHardware::CreateEdgeInfo(builder,
                                                   ir_steps[(unsigned)nn_ir::EdgeExecutionStepType::LOAD_START],
                                                   ir_steps[(unsigned)nn_ir::EdgeExecutionStepType::LOAD_SYNC],
                                                   ir_steps[(unsigned)nn_ir::EdgeExecutionStepType::STORE_START],
                                                   ir_steps[(unsigned)nn_ir::EdgeExecutionStepType::STORE_SYNC],
                                                   ir_mem_infos);
            return IR::CreateEdge(builder,
                                  ir_edge_id,
                                  ir_edge_name,
                                  ir_graph_id,
                                  ir_in_node_id,
                                  ir_out_node_id,
                                  ir_blob_id,
                                  ir_edge_type,
                                  hw_edge_info);
        }
        case nn_ir::EdgeType::CONTROL: {
            Log::IR::E() << "IRExporter::saveEdge() => control edge not support!";
        }
        default: {
            Log::IR::E() << "IRExporter::saveEdge() => unknown edge_type!";
        }
    }
}

flatbuffers::Offset<IR::NNNode::ActivationNode> IRExporter::makeActivation(flatbuffers::FlatBufferBuilder& builder,
                                                                           const nn_ir::ActivationNode&    act_node) {
    auto ir_slope          = act_node.getSlope();
    auto ir_negative_slope = act_node.getNegativeSlope();
    auto ir_min            = act_node.getMin();
    auto ir_max            = act_node.getMax();

    nn_ir::NNIR_Node_Config_Type_ activation_type = act_node.getActivationType();

    IR::NNNode::ActivationType ir_activation_type =
        std::get<IR::NNNode::ActivationType>(nn_ir::parseConfigType(activation_type));
    flatbuffers::Offset<IR::OPNode::ShiftNode> shift_node = 0;
    if (act_node.getShiftNode() != nullptr) {
        shift_node = makeShift(builder, act_node.getShiftNode());
    }

    return IR::NNNode::CreateActivationNode(
        builder, ir_activation_type, ir_slope, ir_negative_slope, ir_min, ir_max, shift_node);
}

flatbuffers::Offset<IR::OPNode::ShiftNode> IRExporter::makeShift(flatbuffers::FlatBufferBuilder& builder,
                                                                 const nn_ir::ShiftNode*         shift_node) {
    if (shift_node == nullptr)
        return 0;

    auto ir_quantization_shift   = builder.CreateVector(shift_node->getQuantizationShift());
    auto ir_multiplication_shift = builder.CreateVector(shift_node->getMultiplicationShift());
    auto ir_activation_shift     = builder.CreateVector(shift_node->getActivationShift());
    auto ir_lut_scale            = builder.CreateVector(shift_node->getLutScale());
    auto ir_lut_bias             = builder.CreateVector(shift_node->getLutBias());
    auto ir_grelu_info           = builder.CreateVector(shift_node->getGreluInfo());

    return IR::OPNode::CreateShiftNode(builder,
                                       ir_quantization_shift,
                                       ir_multiplication_shift,
                                       ir_activation_shift,
                                       ir_lut_scale,
                                       ir_lut_bias,
                                       ir_grelu_info);
}

template <>
flatbuffers::Offset<IR::NnNode>
IRExporter::makeNNNodeIR<nn_ir::NodeType::INPUT>(flatbuffers::FlatBufferBuilder& builder,
                                                 const nn_ir::NNNode&            nn_node) {
    const auto& input_node = static_cast<const nn_ir::InputNode&>(nn_node);
    auto        ir_scale   = input_node.getScale();
    auto        ir_mirror  = input_node.getMirror();
    auto        ir_mean    = builder.CreateVector(input_node.getMean());

    nn_ir::NNIR_Node_Config_Type_ input_type = input_node.getInputType();

    IR::NNNode::InputType ir_input_type = std::get<IR::NNNode::InputType>(nn_ir::parseConfigType(input_type));

    auto ir_input_node = IR::NNNode::CreateInputNode(builder, ir_input_type, ir_mean, ir_scale, ir_mirror);
    return IR::CreateNnNode(builder, IR::NNNode::AnyType_InputNode, ir_input_node.Union());
}

template <>
flatbuffers::Offset<IR::NnNode>
IRExporter::makeNNNodeIR<nn_ir::NodeType::CONVOLUTION>(flatbuffers::FlatBufferBuilder& builder,
                                                       const nn_ir::NNNode&            nn_node) {
    const auto& conv_node              = static_cast<const nn_ir::ConvolutionNode&>(nn_node);
    const auto& kernel_node_parameters = conv_node.getKernelNodeParameters();

    auto activation_node = conv_node.getActivationNode();
    auto shift_node      = conv_node.getShiftNode();
    auto kernel_size     = kernel_node_parameters.getKernelSize();
    auto stride_size     = kernel_node_parameters.getStrideSize();
    auto dilation_size   = kernel_node_parameters.getDilationSize();
    auto padding_size    = kernel_node_parameters.getPaddingSize();

    auto ir_kernel_size    = IR::Type::Dim2(kernel_size.h, kernel_size.w);
    auto ir_stride_size    = IR::Type::Dim2(stride_size.h, stride_size.w);
    auto ir_dilation_size  = IR::Type::Dim2(dilation_size.h, dilation_size.w);
    auto ir_padding_size   = IR::Type::Pad4(padding_size.l, padding_size.r, padding_size.t, padding_size.b);
    auto ir_kernel_blob_id = conv_node.getKernelBlobId();
    auto ir_bias_blob_id   = conv_node.getBiasBlobId();

    flatbuffers::Offset<IR::NNNode::ActivationNode> ir_act_node = 0;
    // activation
    if (activation_node != nullptr) {
        ir_act_node = makeActivation(builder, *activation_node);
    }

    flatbuffers::Offset<IR::OPNode::ShiftNode> ir_shift_node = makeShift(builder, shift_node);

    auto ir_conv_node = IR::NNNode::CreateConvNode(builder,
                                                   ir_act_node,
                                                   &ir_kernel_size,
                                                   &ir_stride_size,
                                                   &ir_dilation_size,
                                                   &ir_padding_size,
                                                   ir_kernel_blob_id,
                                                   ir_bias_blob_id,
                                                   ir_shift_node);

    return IR::CreateNnNode(builder, IR::NNNode::AnyType_ConvNode, ir_conv_node.Union());
}

template <>
flatbuffers::Offset<IR::NnNode>
IRExporter::makeNNNodeIR<nn_ir::NodeType::DECONVOLUTION>(flatbuffers::FlatBufferBuilder& builder,
                                                         const nn_ir::NNNode&            nn_node) {
    const auto& deconv_node            = static_cast<const nn_ir::DeconvolutionNode&>(nn_node);
    const auto& kernel_node_parameters = deconv_node.getKernelNodeParameters();

    auto activation_node = deconv_node.getActivationNode();
    auto shift_node      = deconv_node.getShiftNode();
    auto kernel_size     = kernel_node_parameters.getKernelSize();
    auto stride_size     = kernel_node_parameters.getStrideSize();
    auto dilation_size   = kernel_node_parameters.getDilationSize();
    auto padding_size    = kernel_node_parameters.getPaddingSize();

    auto ir_kernel_size    = IR::Type::Dim2(kernel_size.h, kernel_size.w);
    auto ir_stride_size    = IR::Type::Dim2(stride_size.h, stride_size.w);
    auto ir_dilation_size  = IR::Type::Dim2(dilation_size.h, dilation_size.w);
    auto ir_padding_size   = IR::Type::Pad4(padding_size.l, padding_size.r, padding_size.t, padding_size.b);
    auto ir_kernel_blob_id = deconv_node.getKernelBlobId();
    auto ir_bias_blob_id   = deconv_node.getBiasBlobId();

    flatbuffers::Offset<IR::NNNode::ActivationNode> ir_act_node = 0;
    // activation
    if (activation_node != nullptr) {
        ir_act_node = makeActivation(builder, *activation_node);
    }

    flatbuffers::Offset<IR::OPNode::ShiftNode> ir_shift_node = makeShift(builder, shift_node);

    auto ir_deconv_node = IR::NNNode::CreateDeConvNode(builder,
                                                       ir_act_node,
                                                       &ir_kernel_size,
                                                       &ir_stride_size,
                                                       &ir_dilation_size,
                                                       &ir_padding_size,
                                                       ir_kernel_blob_id,
                                                       ir_bias_blob_id,
                                                       ir_shift_node,
                                                       0,
                                                       0,
                                                       0);

    return IR::CreateNnNode(builder, IR::NNNode::AnyType_DeConvNode, ir_deconv_node.Union());
}

template <>
flatbuffers::Offset<IR::NnNode> IRExporter::makeNNNodeIR<nn_ir::NodeType::POOL>(flatbuffers::FlatBufferBuilder& builder,
                                                                                const nn_ir::NNNode& nn_node) {
    const auto& pool_node              = static_cast<const nn_ir::PoolNode&>(nn_node);
    const auto& kernel_node_parameters = pool_node.getKernelNodeParameters();

    auto kernel_size      = kernel_node_parameters.getKernelSize();
    auto stride_size      = kernel_node_parameters.getStrideSize();
    auto dilation_size    = kernel_node_parameters.getDilationSize();
    auto padding_size     = kernel_node_parameters.getPaddingSize();
    auto ir_kernel_size   = IR::Type::Dim2(kernel_size.h, kernel_size.w);
    auto ir_stride_size   = IR::Type::Dim2(stride_size.h, stride_size.w);
    auto ir_dilation_size = IR::Type::Dim2(dilation_size.h, dilation_size.w);
    auto ir_padding_size  = IR::Type::Pad4(padding_size.l, padding_size.r, padding_size.t, padding_size.b);

    nn_ir::NNIR_Node_Config_Type_ pool_type     = pool_node.getPoolType();
    nn_ir::NNIR_Node_Config_Type_ pad_calc_type = pool_node.getPadCalcType();

    IR::NNNode::PoolType       ir_pool_type = std::get<IR::NNNode::PoolType>(nn_ir::parseConfigType(pool_type));
    IR::NNNode::PadCalculation ir_pad_calc_type =
        std::get<IR::NNNode::PadCalculation>(nn_ir::parseConfigType(pad_calc_type));

    auto ir_pool_node = IR::NNNode::CreatePoolNode(
        builder, ir_pool_type, &ir_kernel_size, &ir_stride_size, &ir_dilation_size, &ir_padding_size, ir_pad_calc_type);
    return IR::CreateNnNode(builder, IR::NNNode::AnyType_PoolNode, ir_pool_node.Union());
}

template <>
flatbuffers::Offset<IR::NnNode>
IRExporter::makeNNNodeIR<nn_ir::NodeType::ACTIVATION>(flatbuffers::FlatBufferBuilder& builder,
                                                      const nn_ir::NNNode&            nn_node) {
    const auto& activation_node = static_cast<const nn_ir::ActivationNode&>(nn_node);

    auto ir_activation_node = makeActivation(builder, activation_node);
    return IR::CreateNnNode(builder, IR::NNNode::AnyType_ActivationNode, ir_activation_node.Union());
}

template <>
flatbuffers::Offset<IR::NnNode>
IRExporter::makeNNNodeIR<nn_ir::NodeType::CONCAT>(flatbuffers::FlatBufferBuilder& builder,
                                                  const nn_ir::NNNode&            nn_node) {
    const auto& concat_node = static_cast<const nn_ir::ConcatNode&>(nn_node);
    const auto  ir_axis     = static_cast<int8_t>(concat_node.getAxis());

    auto ir_concat_node = IR::NNNode::CreateConcatNode(builder, ir_axis);
    return IR::CreateNnNode(builder, IR::NNNode::AnyType_ConcatNode, ir_concat_node.Union());
}

template <>
flatbuffers::Offset<IR::NnNode>
IRExporter::makeNNNodeIR<nn_ir::NodeType::SOFTMAX>(flatbuffers::FlatBufferBuilder& builder,
                                                   const nn_ir::NNNode&            nn_node) {
    const auto& softmax_node         = static_cast<const nn_ir::SoftmaxNode&>(nn_node);
    const auto  ir_axis              = static_cast<int8_t>(softmax_node.getAxis());
    const auto  exp_lut_blob_id      = softmax_node.getExpLUTBlobId();
    const auto  exp_scale            = softmax_node.getExpScale();
    const auto  exp_bias             = softmax_node.getExpBias();
    const auto  softmax_lut_blob_id  = softmax_node.getSoftmaxLUTBlobId();
    const auto  softmax_scale_ex     = softmax_node.getSoftmaxScaleEx();
    const auto  softmax_max_sum_ex   = softmax_node.getSoftmaxMaxSumEx();
    const auto  softmax_max_ex       = softmax_node.getSoftmaxMaxEx();
    const auto  softmax_scale_sum_ex = softmax_node.getSoftmaxScaleSumEx();
    const auto  has_mask             = softmax_node.hasMask();

    auto ir_softmax_node = IR::NNNode::CreateSoftmaxNode(builder,
                                                         ir_axis,
                                                         exp_lut_blob_id,
                                                         exp_scale,
                                                         exp_bias,
                                                         softmax_lut_blob_id,
                                                         softmax_scale_ex,
                                                         softmax_max_ex,
                                                         softmax_scale_sum_ex,
                                                         softmax_max_sum_ex,
                                                         has_mask);
    return IR::CreateNnNode(builder, IR::NNNode::AnyType_SoftmaxNode, ir_softmax_node.Union());
}

template <>
flatbuffers::Offset<IR::NnNode>
IRExporter::makeNNNodeIR<nn_ir::NodeType::BATCHNORM>(flatbuffers::FlatBufferBuilder& builder,
                                                     const nn_ir::NNNode&            nn_node) {
    const auto& bn_node             = static_cast<const nn_ir::BatchNormNode&>(nn_node);
    const auto  ir_use_global_stats = bn_node.getUseGlobalStats();
    const auto  ir_eps              = bn_node.getEps();
    const auto  ir_scale            = bn_node.getScale();
    const auto  ir_axis             = static_cast<int8_t>(bn_node.getAxis());

    flatbuffers::Offset<IR::Type::TypedArray> ir_std_arr;
    flatbuffers::Offset<IR::Type::TypedArray> ir_mean_arr;
    auto                                      ir_f32_arr = builder.CreateVector(std::vector<float>());
    if (bn_node.getStdBuf().size() > 0) {
        ir_f32_arr = builder.CreateVector(bn_node.getStdBuf());
        ir_std_arr = IR::Type::CreateTypedArray(builder, ir_f32_arr, 0, 0, 0);
    }

    if (bn_node.getMeanBuf().size() > 0) {
        ir_f32_arr  = builder.CreateVector(bn_node.getMeanBuf());
        ir_mean_arr = IR::Type::CreateTypedArray(builder, ir_f32_arr, 0, 0, 0);
    }

    auto ir_bn_node = IR::NNNode::CreateBatchNormNode(
        builder, ir_axis, ir_use_global_stats, ir_scale, ir_mean_arr, ir_std_arr, ir_eps);
    return IR::CreateNnNode(builder, IR::NNNode::AnyType_BatchNormNode, ir_bn_node.Union());
}

template <>
flatbuffers::Offset<IR::NnNode>
IRExporter::makeNNNodeIR<nn_ir::NodeType::SCALE>(flatbuffers::FlatBufferBuilder& builder,
                                                 const nn_ir::NNNode&            nn_node) {
    const auto& scale_node = static_cast<const nn_ir::ScaleNode&>(nn_node);
    auto        bias_term  = scale_node.getBiasTerm();

    flatbuffers::Offset<IR::Type::TypedArray> ir_alpha_arr;
    flatbuffers::Offset<IR::Type::TypedArray> ir_beta_arr;
    auto                                      ir_f32_arr = builder.CreateVector(std::vector<float>());
    if (scale_node.getAlphaBuf().size() > 0) {
        ir_f32_arr   = builder.CreateVector(scale_node.getAlphaBuf());
        ir_alpha_arr = IR::Type::CreateTypedArray(builder, ir_f32_arr, 0, 0, 0);
    }

    if (scale_node.getBetaBuf().size() > 0) {
        ir_f32_arr  = builder.CreateVector(scale_node.getBetaBuf());
        ir_beta_arr = IR::Type::CreateTypedArray(builder, ir_f32_arr, 0, 0, 0);
    }

    auto ir_scale_node = IR::NNNode::CreateScaleNode(builder, bias_term, ir_alpha_arr, ir_beta_arr);

    return IR::CreateNnNode(builder, IR::NNNode::AnyType_ScaleNode, ir_scale_node.Union());
}

template <>
flatbuffers::Offset<IR::NnNode>
IRExporter::makeNNNodeIR<nn_ir::NodeType::FULLYCONNECTED>(flatbuffers::FlatBufferBuilder& builder,
                                                          const nn_ir::NNNode&            nn_node) {
    const auto& fully_connected_node = static_cast<const nn_ir::FullyConnectedNode&>(nn_node);
    const auto  activation_node      = fully_connected_node.getActivationNode();
    const auto  shift_node           = fully_connected_node.getShiftNode();
    const auto  ir_axis              = static_cast<int8_t>(fully_connected_node.getAxis());
    const auto  ir_transpose         = fully_connected_node.getTranspose();
    const auto  ir_bias_blob_id      = fully_connected_node.getBiasBlobId();
    const auto  ir_weight_blob_id    = fully_connected_node.getKernelBlobId();

    flatbuffers::Offset<IR::NNNode::ActivationNode> ir_act_node = 0;
    // activation
    if (activation_node != nullptr) {
        ir_act_node = makeActivation(builder, *activation_node);
    }

    flatbuffers::Offset<IR::OPNode::ShiftNode> ir_shift_node = 0;
    if (shift_node != nullptr) {
        auto ir_quantization_shift   = builder.CreateVector(shift_node->getQuantizationShift());
        auto ir_multiplication_shift = builder.CreateVector(shift_node->getMultiplicationShift());
        auto ir_activation_shift     = builder.CreateVector(shift_node->getActivationShift());
        auto ir_lut_scale            = builder.CreateVector(shift_node->getLutScale());
        auto ir_lut_bias             = builder.CreateVector(shift_node->getLutBias());
        auto ir_grelu_info           = builder.CreateVector(shift_node->getGreluInfo());
        ir_shift_node                = IR::OPNode::CreateShiftNode(builder,
                                                    ir_quantization_shift,
                                                    ir_multiplication_shift,
                                                    ir_activation_shift,
                                                    ir_lut_scale,
                                                    ir_lut_bias,
                                                    ir_grelu_info);
    }

    auto ir_fully_connected_node = IR::NNNode::CreateFullyConnectedNode(
        builder, ir_act_node, ir_axis, ir_transpose, ir_weight_blob_id, ir_bias_blob_id, ir_shift_node);
    return IR::CreateNnNode(builder, IR::NNNode::AnyType_FullyConnectedNode, ir_fully_connected_node.Union());
}

template <>
flatbuffers::Offset<IR::NnNode>
IRExporter::makeNNNodeIR<nn_ir::NodeType::ELTWISE>(flatbuffers::FlatBufferBuilder& builder,
                                                   const nn_ir::NNNode&            nn_node) {
    const auto& eltwise_node     = static_cast<const nn_ir::EltwiseNode&>(nn_node);
    auto        stable_prod_grad = eltwise_node.getStableProdGrad();
    auto        shift_node       = eltwise_node.getShiftNode();
    auto        shift_in1_node   = eltwise_node.getShiftIn1Node();
    auto        shift_in2_node   = eltwise_node.getShiftIn2Node();
    uint16_t    multi_scale      = eltwise_node.getMultiScale();

    nn_ir::NNIR_Node_Config_Type_ elt_type    = eltwise_node.getEltType();
    IR::NNNode::EltwiseType       ir_elt_type = std::get<IR::NNNode::EltwiseType>(nn_ir::parseConfigType(elt_type));

    flatbuffers::Offset<IR::OPNode::ShiftNode> ir_shift_node     = makeShift(builder, shift_node);
    flatbuffers::Offset<IR::OPNode::ShiftNode> ir_shift_in1_node = makeShift(builder, shift_in1_node);
    flatbuffers::Offset<IR::OPNode::ShiftNode> ir_shift_in2_node = makeShift(builder, shift_in2_node);

    auto ir_eltwise_node = IR::NNNode::CreateEltwiseNode(
        builder, ir_elt_type, stable_prod_grad, ir_shift_in1_node, ir_shift_in2_node, ir_shift_node, multi_scale);

    return IR::CreateNnNode(builder, IR::NNNode::AnyType_EltwiseNode, ir_eltwise_node.Union());
}

template <>
flatbuffers::Offset<IR::HwNode>
IRExporter::makeHWNodeIR<nn_ir::NodeType::MAAELTWISE>(flatbuffers::FlatBufferBuilder& builder,
                                                      const nn_ir::HWNode&            hw_node) {
    const auto& eltwise_node     = static_cast<const nn_ir::MAAEltwiseNode&>(hw_node);
    auto        stable_prod_grad = eltwise_node.getStableProdGrad();
    auto        shift_node       = eltwise_node.getShiftNode();
    auto        activation_node  = eltwise_node.getActivationNode();

    nn_ir::NNIR_Node_Config_Type_ elt_type = eltwise_node.getEltType();

    IR::NNNode::EltwiseType ir_elt_type = std::get<IR::NNNode::EltwiseType>(nn_ir::parseConfigType(elt_type));

    flatbuffers::Offset<IR::OPNode::ShiftNode> ir_shift_node  = makeShift(builder, shift_node);
    auto                                       kernel_blob_id = eltwise_node.getKernelBlobId();
    auto                                       bias_blob_id   = eltwise_node.getBiasBlobId();

    flatbuffers::Offset<IR::NNNode::ActivationNode> ir_act_node = 0;
    // activation
    if (activation_node != nullptr) {
        ir_act_node = makeActivation(builder, *activation_node);
    }

    auto ir_eltwise_node = IR::HWNode::CreateMAAEltwiseNode(
        builder, ir_elt_type, stable_prod_grad, ir_shift_node, ir_act_node, kernel_blob_id, bias_blob_id);

    return IR::CreateHwNode(builder, IR::HWNode::AnyType_MAAEltwiseNode, ir_eltwise_node.Union());
}

template <>
flatbuffers::Offset<IR::OpNode>
IRExporter::makeOPNodeIR<nn_ir::NodeType::SHIFT>(flatbuffers::FlatBufferBuilder& builder,
                                                 const nn_ir::OPNode&            op_node) {
    const auto& shift_node = static_cast<const nn_ir::ShiftNode&>(op_node);

    auto ir_quantization_shift   = builder.CreateVector(shift_node.getQuantizationShift());
    auto ir_multiplication_shift = builder.CreateVector(shift_node.getMultiplicationShift());
    auto ir_activation_shift     = builder.CreateVector(shift_node.getActivationShift());
    auto ir_lut_scale            = builder.CreateVector(shift_node.getLutScale());
    auto ir_lut_bias             = builder.CreateVector(shift_node.getLutBias());
    auto ir_grelu_info           = builder.CreateVector(shift_node.getGreluInfo());
    auto ir_shift_node           = IR::OPNode::CreateShiftNode(builder,
                                                     ir_quantization_shift,
                                                     ir_multiplication_shift,
                                                     ir_activation_shift,
                                                     ir_lut_scale,
                                                     ir_lut_bias,
                                                     ir_grelu_info);

    return IR::CreateOpNode(builder, IR::OPNode::AnyType_ShiftNode, ir_shift_node.Union());
}

template <>
flatbuffers::Offset<IR::NnNode>
IRExporter::makeNNNodeIR<nn_ir::NodeType::DATAFORMAT>(flatbuffers::FlatBufferBuilder& builder,
                                                      const nn_ir::NNNode&            nn_node) {
    const auto& data_format_node = static_cast<const nn_ir::DataFormatNode&>(nn_node);

    auto                             format_direction = data_format_node.getFormatDirection();
    IR::NNNode::DataFormatConversion ir_format_direction;
    switch (format_direction) {
        case nn_ir::DataFormatConversion::TENSOR2CELL:
            ir_format_direction = IR::NNNode::DataFormatConversion_TENSOR2CELL;
            break;
        case nn_ir::DataFormatConversion::CELL2TENSOR:
            ir_format_direction = IR::NNNode::DataFormatConversion_CELL2TENSOR;
            break;
        default:
            Log::IR::E() << "IRExporter::makeNNNodeIR() => wrong format direction!";
            break;
    }
    auto shape    = data_format_node.getShape();
    auto ir_shape = saveDim4(shape);

    auto ir_data_format_node = IR::NNNode::CreateDataFormatNode(builder, ir_format_direction, &ir_shape);

    return IR::CreateNnNode(builder, IR::NNNode::AnyType_DataFormatNode, ir_data_format_node.Union());
}

template <>
flatbuffers::Offset<IR::NnNode>
IRExporter::makeNNNodeIR<nn_ir::NodeType::RESHAPE>(flatbuffers::FlatBufferBuilder& builder,
                                                   const nn_ir::NNNode&            nn_node) {
    const auto& reshape_node = static_cast<const nn_ir::ReshapeNode&>(nn_node);
    auto        shape        = reshape_node.getShape();

    auto ir_shape        = saveDim4(shape);
    auto ir_reshape_node = IR::NNNode::CreateReshapeNode(builder, &ir_shape);
    return IR::CreateNnNode(builder, IR::NNNode::AnyType_ReshapeNode, ir_reshape_node.Union());
}

template <>
flatbuffers::Offset<IR::NnNode>
IRExporter::makeNNNodeIR<nn_ir::NodeType::PERMUTE>(flatbuffers::FlatBufferBuilder& builder,
                                                   const nn_ir::NNNode&            nn_node) {
    const auto& permute_node  = static_cast<const nn_ir::PermuteNode&>(nn_node);
    auto        permute_order = permute_node.getPermuteOrder();
    auto        in_shape      = permute_node.getInputShape();

    auto ir_permute_order = saveDim4(permute_order);
    auto ir_in_shape      = saveDim4(in_shape);
    auto ir_permute_node  = IR::NNNode::CreatePermuteNode(builder, &ir_permute_order, &ir_in_shape);
    return IR::CreateNnNode(builder, IR::NNNode::AnyType_PermuteNode, ir_permute_node.Union());
}

template <>
flatbuffers::Offset<IR::NnNode>
IRExporter::makeNNNodeIR<nn_ir::NodeType::PRIORBOX>(flatbuffers::FlatBufferBuilder& builder,
                                                    const nn_ir::NNNode&            nn_node) {
    const auto& priorbox_node = static_cast<const nn_ir::PriorBoxNode&>(nn_node);

    auto ir_flip    = priorbox_node.getFlip();
    auto ir_clip    = priorbox_node.getClip();
    auto ir_step_h  = priorbox_node.getStepH();
    auto ir_step_w  = priorbox_node.getStepW();
    auto ir_offset  = priorbox_node.getOffset();
    auto ir_blob_id = priorbox_node.getBlobId();

    flatbuffers::Offset<IR::Type::TypedArray> ir_min_sizes;
    flatbuffers::Offset<IR::Type::TypedArray> ir_max_sizes;
    flatbuffers::Offset<IR::Type::TypedArray> ir_aspect_ratios;
    flatbuffers::Offset<IR::Type::TypedArray> ir_variance;

    auto ir_f32_arr = builder.CreateVector(std::vector<float>());
    if (priorbox_node.getMinSizes().size() > 0) {
        ir_f32_arr   = builder.CreateVector(priorbox_node.getMinSizes());
        ir_min_sizes = IR::Type::CreateTypedArray(builder, ir_f32_arr, 0, 0, 0);
    }
    if (priorbox_node.getMaxSizes().size() > 0) {
        ir_f32_arr   = builder.CreateVector(priorbox_node.getMaxSizes());
        ir_max_sizes = IR::Type::CreateTypedArray(builder, ir_f32_arr, 0, 0, 0);
    }
    if (priorbox_node.getAspectRatios().size() > 0) {
        ir_f32_arr       = builder.CreateVector(priorbox_node.getAspectRatios());
        ir_aspect_ratios = IR::Type::CreateTypedArray(builder, ir_f32_arr, 0, 0, 0);
    }
    if (priorbox_node.getVariance().size() > 0) {
        ir_f32_arr  = builder.CreateVector(priorbox_node.getVariance());
        ir_variance = IR::Type::CreateTypedArray(builder, ir_f32_arr, 0, 0, 0);
    }

    nn_ir::NNIR_Node_Config_Type_ priorbox_type = priorbox_node.getPriorboxType();
    IR::Type::PriorboxType        ir_type = std::get<IR::Type::PriorboxType>(nn_ir::parseConfigType(priorbox_type));

    auto ir_priorbox_node = IR::NNNode::CreatePriorBoxNode(builder,
                                                           ir_min_sizes,
                                                           ir_max_sizes,
                                                           ir_aspect_ratios,
                                                           ir_flip,
                                                           ir_clip,
                                                           ir_variance,
                                                           ir_step_h,
                                                           ir_step_w,
                                                           ir_offset,
                                                           ir_type,
                                                           ir_blob_id);

    return IR::CreateNnNode(builder, IR::NNNode::AnyType_PriorBoxNode, ir_priorbox_node.Union());
}

template <>
flatbuffers::Offset<IR::NnNode>
IRExporter::makeNNNodeIR<nn_ir::NodeType::SLICE>(flatbuffers::FlatBufferBuilder& builder,
                                                 const nn_ir::NNNode&            nn_node) {
    const auto& slice_node = static_cast<const nn_ir::SliceNode&>(nn_node);
    const auto  ir_axis    = static_cast<int8_t>(slice_node.getAxis());

    flatbuffers::Offset<IR::Type::TypedArray> ir_points;

    auto ir_ui8_arr = builder.CreateVector(std::vector<uint8_t>());
    if (slice_node.getPoints().size() > 0) {
        ir_ui8_arr = builder.CreateVector(slice_node.getPoints());
        ir_points  = IR::Type::CreateTypedArray(builder, 0, 0, 0, 0, ir_ui8_arr, 0);
    }

    auto ir_slice_node = IR::NNNode::CreateSliceNode(builder, ir_axis, ir_points);
    return IR::CreateNnNode(builder, IR::NNNode::AnyType_SliceNode, ir_slice_node.Union());
}

template <>
flatbuffers::Offset<IR::NnNode> IRExporter::makeNNNodeIR<nn_ir::NodeType::TILE>(flatbuffers::FlatBufferBuilder& builder,
                                                                                const nn_ir::NNNode& nn_node) {
    const auto& tile_node = static_cast<const nn_ir::TileNode&>(nn_node);
    const auto  ir_axis   = static_cast<int8_t>(tile_node.getAxis());
    const auto  ir_tiles  = static_cast<int32_t>(tile_node.getTiles());

    auto ir_tile_node = IR::NNNode::CreateTileNode(builder, ir_axis, ir_tiles);
    return IR::CreateNnNode(builder, IR::NNNode::AnyType_TileNode, ir_tile_node.Union());
}

template <>
flatbuffers::Offset<IR::NnNode>
IRExporter::makeNNNodeIR<nn_ir::NodeType::MATMUL>(flatbuffers::FlatBufferBuilder& builder,
                                                  const nn_ir::NNNode&            nn_node) {
    auto                                       matmul_node   = cast_if<nn_ir::MatMulNode>(nn_node);
    auto                                       shift_node    = matmul_node->getShiftNode();
    flatbuffers::Offset<IR::OPNode::ShiftNode> ir_shift_node = makeShift(builder, shift_node);

    auto ir_matmul_node = IR::NNNode::CreateMatMulNode(builder, ir_shift_node);
    return IR::CreateNnNode(builder, IR::NNNode::AnyType_MatMulNode, ir_matmul_node.Union());
}

template <>
flatbuffers::Offset<IR::globalNode>
IRExporter::makeGlobalNodeIR<nn_ir::NodeType::GSPLIT>(flatbuffers::FlatBufferBuilder& builder,
                                                      const nn_ir::GlobalNode&        g_node) {
    const auto& gsplit = static_cast<const nn_ir::GlobalSplitNode&>(g_node);

    nn_ir::NNIR_Node_Config_Type_ nnir_partition_mode = gsplit.getPartitionMode();
    auto ir_partition_mode = std::get<IR::GlobalNode::PartitionModeType>(nn_ir::parseConfigType(nnir_partition_mode));

    IR::Type::Dim4 ir_ifm_starts = saveCoordinate(gsplit.getIfmStarts());

    if (gsplit.getSyncType() != nn_ir::SyncType::NONE) {
        nn_ir::NNIR_Node_Config_Type_ nnir_sync_type = gsplit.getSyncType();
        auto ir_sync_type = std::get<IR::GlobalNode::SyncType>(nn_ir::parseConfigType(nnir_sync_type));
        nn_ir::NNIR_Node_Config_Type_ nnir_sig_type = gsplit.getSigType();
        auto ir_sig_type  = std::get<IR::GlobalNode::SigType>(nn_ir::parseConfigType(nnir_sig_type));
        auto ir_sync_node = IR::GlobalNode::CreateGlobalSyncNode(builder, gsplit.getUId(), ir_sync_type, ir_sig_type);

        auto ir_gsplit = IR::GlobalNode::CreateGlobalSplitNode(
            builder, gsplit.getUId(), ir_partition_mode, ir_sync_node, &ir_ifm_starts);
        return IR::CreateglobalNode(builder, IR::GlobalNode::AnyType_GlobalSplitNode, ir_gsplit.Union());
    } else {
        auto ir_gsplit =
            IR::GlobalNode::CreateGlobalSplitNode(builder, gsplit.getUId(), ir_partition_mode, 0, &ir_ifm_starts);
        return IR::CreateglobalNode(builder, IR::GlobalNode::AnyType_GlobalSplitNode, ir_gsplit.Union());
    }
}

template <>
flatbuffers::Offset<IR::globalNode>
IRExporter::makeGlobalNodeIR<nn_ir::NodeType::GCONCAT>(flatbuffers::FlatBufferBuilder& builder,
                                                       const nn_ir::GlobalNode&        g_node) {
    const auto& gconcat = static_cast<const nn_ir::GlobalConcatNode&>(g_node);

    nn_ir::NNIR_Node_Config_Type_ nnir_concat_axis = gconcat.getConcatAxis();
    auto ir_concat_axis = std::get<IR::GlobalNode::GlobalConcatAxis>(nn_ir::parseConfigType(nnir_concat_axis));
    nn_ir::NNIR_Node_Config_Type_ nnir_concat_type = gconcat.getConcatType();
    auto ir_concat_type = std::get<IR::GlobalNode::GlobalConcatType>(nn_ir::parseConfigType(nnir_concat_type));

    IR::Type::Dim4 ir_ofm_starts = saveCoordinate(gconcat.getOfmStarts());

    if (gconcat.getSyncType() != nn_ir::SyncType::NONE) {
        nn_ir::NNIR_Node_Config_Type_ nnir_sync_type = gconcat.getSyncType();
        auto ir_sync_type = std::get<IR::GlobalNode::SyncType>(nn_ir::parseConfigType(nnir_sync_type));
        nn_ir::NNIR_Node_Config_Type_ nnir_sig_type = gconcat.getSigType();
        auto ir_sig_type  = std::get<IR::GlobalNode::SigType>(nn_ir::parseConfigType(nnir_sig_type));
        auto ir_sync_node = IR::GlobalNode::CreateGlobalSyncNode(builder, gconcat.getUId(), ir_sync_type, ir_sig_type);
        auto ir_gconcat   = IR::GlobalNode::CreateGlobalConcatNode(
            builder, gconcat.getUId(), ir_concat_type, ir_concat_axis, ir_sync_node, &ir_ofm_starts);
        return IR::CreateglobalNode(builder, IR::GlobalNode::AnyType_GlobalConcatNode, ir_gconcat.Union());
    } else {
        auto ir_gconcat = IR::GlobalNode::CreateGlobalConcatNode(
            builder, gconcat.getUId(), ir_concat_type, ir_concat_axis, 0, &ir_ofm_starts);
        return IR::CreateglobalNode(builder, IR::GlobalNode::AnyType_GlobalConcatNode, ir_gconcat.Union());
    }
}

template <>
flatbuffers::Offset<IR::globalNode>
IRExporter::makeGlobalNodeIR<nn_ir::NodeType::GSYNC>(flatbuffers::FlatBufferBuilder& builder,
                                                     const nn_ir::GlobalNode&        g_node) {
    const auto& gsync = static_cast<const nn_ir::GlobalSyncNode&>(g_node);

    if (gsync.getSyncType() != nn_ir::SyncType::NONE) {
        nn_ir::NNIR_Node_Config_Type_ nnir_sync_type = gsync.getSyncType();
        auto ir_sync_type = std::get<IR::GlobalNode::SyncType>(nn_ir::parseConfigType(nnir_sync_type));
        nn_ir::NNIR_Node_Config_Type_ nnir_sig_type = gsync.getSigType();
        auto ir_sig_type  = std::get<IR::GlobalNode::SigType>(nn_ir::parseConfigType(nnir_sig_type));
        auto ir_sync_node = IR::GlobalNode::CreateGlobalSyncNode(builder, gsync.getUId(), ir_sync_type, ir_sig_type);
        return IR::CreateglobalNode(builder, IR::GlobalNode::AnyType_GlobalSyncNode, ir_sync_node.Union());
    } else {
        Log::IR::E() << "global sync node should not with SyncType::NONE";
    }
}

template <>
flatbuffers::Offset<IR::NnNode>
IRExporter::makeNNNodeIR<nn_ir::NodeType::DUMMY>(flatbuffers::FlatBufferBuilder& builder,
                                                 const nn_ir::NNNode&            nn_node) {
    auto ir_dummy_node = IR::NNNode::CreateDummyNode(builder);
    return IR::CreateNnNode(builder, IR::NNNode::AnyType_DummyNode, ir_dummy_node.Union());
}

template <>
flatbuffers::Offset<IR::NnNode> IRExporter::makeNNNodeIR<nn_ir::NodeType::COPY>(flatbuffers::FlatBufferBuilder& builder,
                                                                                const nn_ir::NNNode& nn_node) {
    auto copy_node = IR::NNNode::CreateCopyNode(builder);
    return IR::CreateNnNode(builder, IR::NNNode::AnyType_CopyNode, copy_node.Union());
}

template <>
flatbuffers::Offset<IR::vNode>
IRExporter::makeVNodeIR<nn_ir::NodeType::VSPLIT>(flatbuffers::FlatBufferBuilder&  builder,
                                                 const nn_compiler::nn_ir::VNode& v_node) {
    const auto& split_node   = static_cast<const nn_ir::VSplitNode&>(v_node);
    auto        num_tiles    = split_node.getNumTiles();
    auto        ir_num_tiles = IR::Type::TileNumbers(num_tiles.n, num_tiles.h, num_tiles.w, num_tiles.c);

    nn_ir::NNIR_Node_Config_Type_ nnir_tiling_shceme = nn_ir::getTilingScheme(split_node.getNumTiles());
    auto ir_tiling_scheme = std::get<IR::Type::TilingSchemeType>(nn_ir::parseConfigType(nnir_tiling_shceme));

    std::vector<int8_t> ir_tiling_order_vector;
    auto                ir_tiling_order = builder.CreateVector(ir_tiling_order_vector);

    std::vector<flatbuffers::Offset<IR::Type::TilePosition>> tile_positions;
    for (const auto& tile_info : split_node.getTileInfos()) {
        auto ir_tile_numbers = IR::Type::TileNumbers(
            tile_info.position.n, tile_info.position.h, tile_info.position.w, tile_info.position.c);

        // FIXME: Incorrect order, use saveDim4()
        auto ir_first_value_coord = IR::Type::Dim4(tile_info.first_value_coord.w,
                                                   tile_info.first_value_coord.h,
                                                   tile_info.first_value_coord.c,
                                                   tile_info.first_value_coord.n);

        tile_positions.emplace_back(
            IR::Type::CreateTilePosition(builder, tile_info.node_id, &ir_tile_numbers, &ir_first_value_coord));
    }
    auto ir_tile_positions = builder.CreateVector(tile_positions);

    auto ir_split_node =
        IR::VNode::CreateVSplitNode(builder, &ir_num_tiles, ir_tiling_scheme, ir_tiling_order, ir_tile_positions);

    return IR::CreatevNode(builder, IR::VNode::AnyType_VSplitNode, ir_split_node.Union());
}

template <>
flatbuffers::Offset<IR::vNode>
IRExporter::makeVNodeIR<nn_ir::NodeType::VCONCAT>(flatbuffers::FlatBufferBuilder&  builder,
                                                  const nn_compiler::nn_ir::VNode& v_node) {
    const auto& concat_node    = static_cast<const nn_ir::VConcatNode&>(v_node);
    auto        ir_concat_node = IR::VNode::CreateVConcatNode(builder, concat_node.getVsplitNodeId());
    return IR::CreatevNode(builder, IR::VNode::AnyType_VConcatNode, ir_concat_node.Union());
}

template <typename q_node_type>
std::vector<flatbuffers::Offset<IR::Type::TypedArray>> createQNodeArrays(q_node_type                     qnode,
                                                                         flatbuffers::FlatBufferBuilder& builder) {
    flatbuffers::Offset<IR::Type::TypedArray>              ir_output_scales;
    flatbuffers::Offset<IR::Type::TypedArray>              ir_output_zp;
    flatbuffers::Offset<IR::Type::TypedArray>              ir_output_frac_len;
    std::vector<flatbuffers::Offset<IR::Type::TypedArray>> typed_arr_vec;

    auto ir_f32_arr = builder.CreateVector(std::vector<float>());
    auto ir_i32_arr = builder.CreateVector(std::vector<int32_t>());
    auto ir_i8_arr  = builder.CreateVector(std::vector<int8_t>());

    if (!qnode.getScale().empty()) {
        ir_f32_arr       = builder.CreateVector(qnode.getScale());
        ir_output_scales = IR::Type::CreateTypedArray(builder, ir_f32_arr, 0, 0, 0);
        typed_arr_vec.push_back(ir_output_scales);
    }
    if (!qnode.getZeroPoint().empty()) {
        ir_i32_arr   = builder.CreateVector(qnode.getZeroPoint());
        ir_output_zp = IR::Type::CreateTypedArray(builder, 0, ir_i32_arr, 0, 0);
        typed_arr_vec.push_back(ir_output_zp);
    }
    if (!qnode.getFracLen().empty()) {
        ir_i8_arr          = builder.CreateVector(qnode.getFracLen());
        ir_output_frac_len = IR::Type::CreateTypedArray(builder, 0, 0, 0, ir_i8_arr, 0);
        typed_arr_vec.push_back(ir_output_frac_len);
    }
    return typed_arr_vec;
}

template <>
flatbuffers::Offset<IR::qNode>
IRExporter::makeQNodeIR<nn_ir::NodeType::QUANT>(flatbuffers::FlatBufferBuilder&  builder,
                                                const nn_compiler::nn_ir::QNode& q_node) {
    const auto& quant_node    = static_cast<const nn_ir::QuantNode&>(q_node);
    auto        quant_type    = quant_node.getQuantType();
    auto        ir_quant_type = static_cast<IR::Type::QuantType>(quant_type);

    auto params_vec = createQNodeArrays<nn_ir::QuantNode>(quant_node, builder);

    if (quant_type == nn_ir::QuantType::ASYMMETRIC) {
        auto ir_asym_quant_param = IR::Type::CreateAsymQuantParam(builder, params_vec[0], params_vec[1]);
        auto ir_quant_node       = IR::QNode::CreateQuantNode(builder,
                                                        ir_quant_type,
                                                        IR::Type::QuantLevelType_LAYERWISE,
                                                        IR::Type::AnyQuantParam_AsymQuantParam,
                                                        ir_asym_quant_param.Union());

        return IR::CreateqNode(builder, IR::QNode::AnyType_QuantNode, ir_quant_node.Union());
    } else {
        auto ir_symm_quant_param = IR::Type::CreateSymmQuantParam(builder, params_vec[0]);
        auto ir_quant_node       = IR::QNode::CreateQuantNode(builder,
                                                        ir_quant_type,
                                                        IR::Type::QuantLevelType_LAYERWISE,
                                                        IR::Type::AnyQuantParam_SymmQuantParam,
                                                        ir_symm_quant_param.Union());

        return IR::CreateqNode(builder, IR::QNode::AnyType_QuantNode, ir_quant_node.Union());
    }
}

template <>
flatbuffers::Offset<IR::qNode>
IRExporter::makeQNodeIR<nn_ir::NodeType::DEQUANT>(flatbuffers::FlatBufferBuilder&  builder,
                                                  const nn_compiler::nn_ir::QNode& q_node) {
    const auto& dequant_node  = static_cast<const nn_ir::DequantNode&>(q_node);
    auto        quant_type    = dequant_node.getQuantType();
    auto        ir_quant_type = static_cast<IR::Type::QuantType>(quant_type);

    auto params_vec = createQNodeArrays<nn_ir::DequantNode>(dequant_node, builder);

    if (quant_type == nn_ir::QuantType::ASYMMETRIC) {
        auto ir_asym_quant_param = IR::Type::CreateAsymQuantParam(builder, params_vec[0], params_vec[1]);
        auto ir_dequant_node     = IR::QNode::CreateDequantNode(builder,
                                                            ir_quant_type,
                                                            IR::Type::QuantLevelType_LAYERWISE,
                                                            IR::Type::AnyQuantParam_AsymQuantParam,
                                                            ir_asym_quant_param.Union());

        return IR::CreateqNode(builder, IR::QNode::AnyType_DequantNode, ir_dequant_node.Union());
    } else {
        auto ir_symm_quant_param = IR::Type::CreateSymmQuantParam(builder, params_vec[0]);
        auto ir_dequant_node     = IR::QNode::CreateDequantNode(builder,
                                                            ir_quant_type,
                                                            IR::Type::QuantLevelType_LAYERWISE,
                                                            IR::Type::AnyQuantParam_SymmQuantParam,
                                                            ir_symm_quant_param.Union());

        return IR::CreateqNode(builder, IR::QNode::AnyType_DequantNode, ir_dequant_node.Union());
    }
}

/**
 * @brief.      save IR from execution step
 * @details.    This function creates IR::Node instance from nn_ir::ExecutionStep
 */
flatbuffers::Offset<IR::TargetHardware::ExecutionStep> IRExporter::saveStep(flatbuffers::FlatBufferBuilder& builder,
                                                                            const nn_ir::ExecutionStep&     step) {
    auto id = step.getId();

    auto save_instrs = [this, &step, &builder]() {
        std::vector<flatbuffers::Offset<IR::TargetHardware::Instruction>> instrs;
        for (const auto& instr : step.getInstructions()) {
            instrs.push_back(saveInstr(builder, *instr));
        }
        return builder.CreateVector(instrs);
    };

    // save load start instructions
    flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<IR::TargetHardware::Instruction>>> instrs(
        save_instrs());

    if (auto node_step = cast_if<nn_ir::NodeExecutionStep>(step)) {
        auto node_id = node_step->getNodeId();

        nn_ir::NNIR_Node_Config_Type_ nnir_step_type = node_step->getNodeStepType();
        auto                          ir_step_type =
            std::get<IR::TargetHardware::Type::NodeExecutionType>(nn_ir::parseConfigType(nnir_step_type));
        auto ir_node_step = IR::TargetHardware::CreateNodeExecutionStep(builder, node_id, ir_step_type);
        return IR::TargetHardware::CreateExecutionStep(
            builder, id, instrs, IR::TargetHardware::AnyExecutionStep_NodeExecutionStep, ir_node_step.Union());
    } else if (auto edge_step = cast_if<nn_ir::EdgeExecutionStep>(step)) {
        auto edge_id = edge_step->getEdgeId();

        nn_ir::NNIR_Node_Config_Type_ nnir_step_type = edge_step->getEdgeStepType();
        auto                          ir_step_type =
            std::get<IR::TargetHardware::Type::EdgeExecutionType>(nn_ir::parseConfigType(nnir_step_type));
        auto ir_edge_step = IR::TargetHardware::CreateEdgeExecutionStep(builder, edge_id, ir_step_type);
        return IR::TargetHardware::CreateExecutionStep(
            builder, id, instrs, IR::TargetHardware::AnyExecutionStep_EdgeExecutionStep, ir_edge_step.Union());
    } else {
        Log::IR::E() << "IRExporter::saveStep => wrong execution step type!";
    }
}

/**
 * @brief.      save IR from instruction
 * @details.    This function creates IR::Node instance from nn_ir::Instruction
 */
flatbuffers::Offset<IR::TargetHardware::Instruction> IRExporter::saveInstr(flatbuffers::FlatBufferBuilder& builder,
                                                                           const nn_ir::Instruction&       instr) {
    auto id = instr.getId();

    if (isa<nn_ir::ComputeInstruction>(instr)) {
        flatbuffers::Offset<IR::TargetHardware::ComputeInstr> ir_compute_instr;
        if (isa<nn_ir::ExecuteStartInstruction>(instr)) {
            // const auto& start_instr    = = cast<nn_ir::ExecuteStartInstruction>(instr);
            auto ir_start_instr = IR::TargetHardware::ComputeInstruction::CreateExecuteStartInstr(builder);
            ir_compute_instr    = IR::TargetHardware::CreateComputeInstr(
                builder, IR::TargetHardware::ComputeInstruction::AnyType_ExecuteStartInstr, ir_start_instr.Union());
        } else if (auto* sync_instr = cast_if<const nn_ir::ExecuteSyncInstruction>(instr)) {
            auto start_id      = sync_instr->getStartId();
            auto ir_sync_instr = IR::TargetHardware::ComputeInstruction::CreateExecuteSyncInstr(builder, start_id);
            ir_compute_instr   = IR::TargetHardware::CreateComputeInstr(
                builder, IR::TargetHardware::ComputeInstruction::AnyType_ExecuteSyncInstr, ir_sync_instr.Union());
        } else {
            Log::IR::E() << "IRExporter::saveStep => wrong compute instruction type!";
        }
        return IR::TargetHardware::CreateInstruction(
            builder, id, IR::TargetHardware::AnyInstr_ComputeInstr, ir_compute_instr.Union());
    } else if (isa<nn_ir::MemoryInstruction>(instr)) {
        flatbuffers::Offset<IR::TargetHardware::MemoryInstr> ir_memory_instr;
        if (auto* start_instr = cast_if<const nn_ir::DMAStartInstruction>(instr)) {
            auto dma_ch_id = start_instr->getDMAChannelId();
            auto dir       = start_instr->getDirection();

            nn_ir::NNIR_Node_Config_Type_ nnir_data_type = start_instr->getDataType();
            auto                          ir_data_type =
                std::get<IR::TargetHardware::Type::MemoryDataType>(nn_ir::parseConfigType(nnir_data_type));
            IR::TargetHardware::Type::MemoryType                 src_type, dst_type;
            uint32_t                                             src_id, dst_id;
            IR::TargetHardware::MemoryInstruction::DirectionType ir_dir;

            if (dir == nn_ir::DMADirection::DRAM2SRAM) {
                src_type = IR::TargetHardware::Type::MemoryType_DRAM;
                src_id   = 0;
                dst_type = IR::TargetHardware::Type::MemoryType_SRAM;
                dst_id   = start_instr->getSramId();
                ir_dir   = IR::TargetHardware::MemoryInstruction::DirectionType_DRAM2SRAM;
            } else if (dir == nn_ir::DMADirection::DRAM2FIFO) {
                src_type = IR::TargetHardware::Type::MemoryType_DRAM;
                src_id   = 0;
                dst_type = IR::TargetHardware::Type::MemoryType_FIFO;
                dst_id   = start_instr->getSramId();
                ir_dir   = IR::TargetHardware::MemoryInstruction::DirectionType_DRAM2FIFO;
            } else {
                src_type = IR::TargetHardware::Type::MemoryType_SRAM;
                src_id   = start_instr->getSramId();
                dst_type = IR::TargetHardware::Type::MemoryType_DRAM;
                dst_id   = 0;
                ir_dir   = IR::TargetHardware::MemoryInstruction::DirectionType_SRAM2DRAM;
            }

            flatbuffers::Offset<IR::TargetHardware::Type::DataLayout> src_layout =
                saveDataLayout(builder, start_instr->getSrcLayout());
            flatbuffers::Offset<IR::TargetHardware::Type::MemoryInfo> src_mem_info =
                IR::TargetHardware::Type::CreateMemoryInfo(builder,
                                                           src_type,
                                                           ir_data_type,
                                                           src_id,
                                                           start_instr->getSize(),
                                                           start_instr->getSrcAddr(),
                                                           src_layout);
            flatbuffers::Offset<IR::TargetHardware::Type::DataLayout> dst_layout =
                saveDataLayout(builder, start_instr->getDstLayout());
            flatbuffers::Offset<IR::TargetHardware::Type::MemoryInfo> dst_mem_info =
                IR::TargetHardware::Type::CreateMemoryInfo(builder,
                                                           dst_type,
                                                           ir_data_type,
                                                           dst_id,
                                                           start_instr->getSize(),
                                                           start_instr->getDstAddr(),
                                                           dst_layout);
            auto ir_start_instr = IR::TargetHardware::MemoryInstruction::CreateDMAStartInstr(
                builder, ir_dir, dma_ch_id, src_mem_info, dst_mem_info);
            ir_memory_instr = IR::TargetHardware::CreateMemoryInstr(
                builder, IR::TargetHardware::MemoryInstruction::AnyType_DMAStartInstr, ir_start_instr.Union());
        } else if (auto* sync_instr = cast_if<const nn_ir::DMASyncInstruction>(instr)) {
            auto start_id      = sync_instr->getStartId();
            auto ir_sync_instr = IR::TargetHardware::MemoryInstruction::CreateDMASyncInstr(builder, start_id);
            ir_memory_instr    = IR::TargetHardware::CreateMemoryInstr(
                builder, IR::TargetHardware::MemoryInstruction::AnyType_DMASyncInstr, ir_sync_instr.Union());
        } else {
            Log::IR::E() << "IRExporter::saveStep => wrong memory instruction type!";
        }
        return IR::TargetHardware::CreateInstruction(
            builder, id, IR::TargetHardware::AnyInstr_MemoryInstr, ir_memory_instr.Union());
    } else if (isa<nn_ir::MiscInstruction>(instr)) {
        flatbuffers::Offset<IR::TargetHardware::MiscInstr> ir_misc_instr;
        if (auto* send_instr = cast_if<const nn_ir::SignalSendInstruction>(instr)) {
            auto ir_send_instr =
                IR::TargetHardware::MiscInstruction::CreateSigSendInstr(builder, send_instr->getDMAChannelId());
            ir_misc_instr = IR::TargetHardware::CreateMiscInstr(
                builder, IR::TargetHardware::MiscInstruction::AnyType_SigSendInstr, ir_send_instr.Union());
        } else if (auto* wait_instr = cast_if<const nn_ir::SignalWaitInstruction>(instr)) {
            auto                                              send_ids = wait_instr->getSendIds();
            flatbuffers::Offset<flatbuffers::Vector<int32_t>> ir_send_ids;
            if (send_ids.size() > 0) {
                ir_send_ids = builder.CreateVector(send_ids);
            }
            auto ir_wait_instr = IR::TargetHardware::MiscInstruction::CreateSigWaitInstr(builder, ir_send_ids);
            ir_misc_instr      = IR::TargetHardware::CreateMiscInstr(
                builder, IR::TargetHardware::MiscInstruction::AnyType_SigWaitInstr, ir_wait_instr.Union());
        } else if (isa<nn_ir::VsyncInstruction>(instr)) {
            return IR::TargetHardware::CreateInstruction(builder, id, IR::TargetHardware::AnyInstr_VSyncInstr);
        } else {
            Log::IR::E() << "IRExporter::saveStep => wrong misc instruction type!";
        }
        return IR::TargetHardware::CreateInstruction(
            builder, id, IR::TargetHardware::AnyInstr_MiscInstr, ir_misc_instr.Union());
    } else {
        Log::IR::E() << "IRExporter::saveStep => wrong instruction type!";
    }
}

/**
 * @brief.      Constructor of IRExporter.
 * @details.    This function constructs IRExporter
 * @param[in].
 * @param[out].
 * @returns.
 */
IRExporter::IRExporter() {
    nn_node_ir_make_func_map_ = {
        {nn_ir::NodeType::INPUT, &IRExporter::makeNNNodeIR<nn_ir::NodeType::INPUT>},
        {nn_ir::NodeType::CONVOLUTION, &IRExporter::makeNNNodeIR<nn_ir::NodeType::CONVOLUTION>},
        {nn_ir::NodeType::POOL, &IRExporter::makeNNNodeIR<nn_ir::NodeType::POOL>},
        {nn_ir::NodeType::ACTIVATION, &IRExporter::makeNNNodeIR<nn_ir::NodeType::ACTIVATION>},
        {nn_ir::NodeType::CONCAT, &IRExporter::makeNNNodeIR<nn_ir::NodeType::CONCAT>},
        {nn_ir::NodeType::SOFTMAX, &IRExporter::makeNNNodeIR<nn_ir::NodeType::SOFTMAX>},
        {nn_ir::NodeType::FULLYCONNECTED, &IRExporter::makeNNNodeIR<nn_ir::NodeType::FULLYCONNECTED>},
        {nn_ir::NodeType::BATCHNORM, &IRExporter::makeNNNodeIR<nn_ir::NodeType::BATCHNORM>},
        {nn_ir::NodeType::SCALE, &IRExporter::makeNNNodeIR<nn_ir::NodeType::SCALE>},
        {nn_ir::NodeType::ELTWISE, &IRExporter::makeNNNodeIR<nn_ir::NodeType::ELTWISE>},
        {nn_ir::NodeType::DECONVOLUTION, &IRExporter::makeNNNodeIR<nn_ir::NodeType::DECONVOLUTION>},
        {nn_ir::NodeType::RESHAPE, &IRExporter::makeNNNodeIR<nn_ir::NodeType::RESHAPE>},
        {nn_ir::NodeType::DATAFORMAT, &IRExporter::makeNNNodeIR<nn_ir::NodeType::DATAFORMAT>},
        {nn_ir::NodeType::PERMUTE, &IRExporter::makeNNNodeIR<nn_ir::NodeType::PERMUTE>},
        {nn_ir::NodeType::PRIORBOX, &IRExporter::makeNNNodeIR<nn_ir::NodeType::PRIORBOX>},
        {nn_ir::NodeType::SLICE, &IRExporter::makeNNNodeIR<nn_ir::NodeType::SLICE>},
        {nn_ir::NodeType::TILE, &IRExporter::makeNNNodeIR<nn_ir::NodeType::TILE>},
        {nn_ir::NodeType::MATMUL, &IRExporter::makeNNNodeIR<nn_ir::NodeType::MATMUL>},
        {nn_ir::NodeType::DUMMY, &IRExporter::makeNNNodeIR<nn_ir::NodeType::DUMMY>},
        {nn_ir::NodeType::COPY, &IRExporter::makeNNNodeIR<nn_ir::NodeType::COPY>}};

    op_node_ir_make_func_map_ = {
        {nn_ir::NodeType::SHIFT, &IRExporter::makeOPNodeIR<nn_ir::NodeType::SHIFT>},
    };

    g_node_ir_make_func_map_ = {
        {nn_ir::NodeType::GSPLIT, &IRExporter::makeGlobalNodeIR<nn_ir::NodeType::GSPLIT>},
        {nn_ir::NodeType::GCONCAT, &IRExporter::makeGlobalNodeIR<nn_ir::NodeType::GCONCAT>},
        {nn_ir::NodeType::GSYNC, &IRExporter::makeGlobalNodeIR<nn_ir::NodeType::GSYNC>},
    };

    v_node_ir_make_func_map_ = {
        {nn_ir::NodeType::VSPLIT, &IRExporter::makeVNodeIR<nn_ir::NodeType::VSPLIT>},
        {nn_ir::NodeType::VCONCAT, &IRExporter::makeVNodeIR<nn_ir::NodeType::VCONCAT>},
    };

    q_node_ir_make_func_map_ = {
        {nn_ir::NodeType::QUANT, &IRExporter::makeQNodeIR<nn_ir::NodeType::QUANT>},
        {nn_ir::NodeType::DEQUANT, &IRExporter::makeQNodeIR<nn_ir::NodeType::DEQUANT>},
    };

    hw_node_ir_make_func_map_ = {
        {nn_ir::NodeType::MAAELTWISE, &IRExporter::makeHWNodeIR<nn_ir::NodeType::MAAELTWISE>},
    };
}

} // namespace nn_compiler
