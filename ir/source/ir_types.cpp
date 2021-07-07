/*
 * Copyright (C) 2019 Samsung Electronics Co. LTD
 *
 * This software is proprietary of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#include "ir/include/ir_types.hpp"

namespace nn_compiler {
namespace nn_ir {

// The following functions return respective strides in bytes.
// TODO(hee-woo-nam): Current cell_unit usage is a hardcoded rule
// for HxW == 1x1 cellformat. Make it universal.
uint32_t DataLayout::calcColStride() const {
    // Catch uninitialized layouts early
    Log::IR::E_IF(byte_order == nn_ir::PixelByteOrder::INVALID_ORDER)
        << "Use of uninitialized byte order in DataLayout";
    uint32_t stride_in_bytes = cell_unit.h * cell_unit.c;
    if (bpp > 1 && byte_order == nn_ir::PixelByteOrder::LITTLE_ENDIAN_ORDER) {
        stride_in_bytes *= bpp;
    }
    return stride_in_bytes + gap.w;
}

uint32_t DataLayout::calcRowStride() const { return calcColStride() * total_dim.w + gap.h; }

// In our cell format channelStride is somewhat fictional, we can't jump over arbitrary number of channels,
// but it appears valid on the border of cells. I. e. it's still possible to calculate channel offset as
// channel_num * layout.calcChannelStride(), given that channel_num is a multiple of layout.cell_unit.c.
// It also happens to be equal to slice stride in cells, so can be used in both ways.
uint32_t DataLayout::calcChannelStride() const {
    // Catch uninitialized layouts early
    Log::IR::E_IF(!cell_unit.isValid()) << "Use of uninitialized DataLayout";
    return (calcRowStride() * total_dim.h + gap.c) / cell_unit.c;
}

MEMORY_SIZE_T DataLayout::calcSizeInBytes(nn_type::BlobUnitSize blob_unit_size) const {
    // This is size of the whole 4D tensor in elements, including all the gaps but N
    uint32_t n_elements = calcChannelStride() * total_dim.c * total_dim.n;
    if (bpp > 1 && byte_order == nn_ir::PixelByteOrder::LITTLE_ENDIAN_ORDER) {
        n_elements /= bpp;
    }
    // gap.n is number of bytes, reserved after the whole tensor. Not between N dimensions!
    // One of reasons for this is that a 4D tensor should be reshape-able into 3D because
    // DMA understands only 3D tensors. 4D is currently used for kernels.
    return getSizeInBytes(n_elements, blob_unit_size) + gap.n;
}

BIT_WIDTH_T getBitWidthByType(DataType type) {
    switch (type) {
        case DataType::FLOAT64:
        case DataType::INT64:
            return 64;

        case DataType::FLOAT32:
        case DataType::INT32:
        case DataType::UINT32:
            return 32;

        case DataType::FLOAT16:
        case DataType::INT16:
        case DataType::UINT16:
            return 16;

        case DataType::INT8:
        case DataType::UINT8:
            return 8;

        case DataType::INT4:
        case DataType::UINT4:
            return 4;

        default:
            Log::IR::E() << "Unknown data type";
            return 0;
    }
}

bool isSignedByType(DataType type) {
    switch (type) {
        case DataType::FLOAT64:
        case DataType::FLOAT32:
        case DataType::FLOAT16:
        case DataType::INT64:
        case DataType::INT32:
        case DataType::INT16:
        case DataType::INT8:
        case DataType::INT4:
            return true;
        case DataType::UINT32:
        case DataType::UINT16:
        case DataType::UINT8:
        case DataType::UINT4:
            return false;
        default:
            Log::IR::E() << "Unknown data type";
            return 0;
    }
}

void printMemInfo(const std::vector<MemoryInfo>& mem_info) {
    for (size_t i = 0; i < mem_info.size(); i++) {
        Log::IR::D() << "* MemInfo[" << i << "]: " << mem_info[i];
    }
}

NNIR_Node_Param_ parseParam(IR_Node_Param_ type) {
    NNIR_Node_Param_ nnir_param;
    std::visit(overloaded{[&nnir_param](const IR::Type::Dim2* dim2) {
                              nn_ir::Shape2D nnir_dim2 = {{.h = dim2->h(), .w = dim2->w()}};
                              nnir_param               = nnir_dim2;
                          },
                          [&nnir_param](const IR::Type::Dim4* dim4) {
                              nn_ir::Shape4D nnir_dim4 = {{
                                  .n = dim4->n(),
                                  .c = dim4->c(),
                                  .h = dim4->h(),
                                  .w = dim4->w(),
                              }};
                              nnir_param               = nnir_dim4;
                          },
                          [&nnir_param](const IR::Type::Pad4* pad4) {
                              nn_ir::Pad4 nnir_pad4 = {.t = pad4->t(), .b = pad4->b(), .l = pad4->l(), .r = pad4->r()};
                              nnir_param            = nnir_pad4;
                          },
                          [&nnir_param](const IR::Type::TilePosition* tile_pos) {
                              nn_ir::TileInfo nnir_tile_info = {
                                  .node_id           = tile_pos->node_id(),
                                  .position          = {{.n = tile_pos->tile_index()->n(),
                                                .c = tile_pos->tile_index()->c(),
                                                .h = tile_pos->tile_index()->h(),
                                                .w = tile_pos->tile_index()->w()}},
                                  .first_value_coord = {{.n = tile_pos->first_value_coord()->n(),
                                                         .c = tile_pos->first_value_coord()->c(),
                                                         .h = tile_pos->first_value_coord()->h(),
                                                         .w = tile_pos->first_value_coord()->w()}}};
                              nnir_param = nnir_tile_info;
                          },
                          [&nnir_param](const IR::Type::TileNumbers* tile_nums) {
                              nn_ir::TileNumbers nnir_tile_nums = {
                                  {.n = tile_nums->n(), .c = tile_nums->c(), .h = tile_nums->h(), .w = tile_nums->w()}};
                              nnir_param = nnir_tile_nums;
                          }},
               type);
    return nnir_param;
} // namespace nn_ir

NNIR_Node_Config_Type_ parseConfigType(IR_Node_Config_Type_& type) {
    NNIR_Node_Config_Type_ nnir_type;
    std::visit(
        [&nnir_type](auto&& args) {
            using T = std::decay_t<decltype(args)>;
            if constexpr (std::is_same_v<T, IR::NNNode::InputType>) {
                nnir_type = nn_ir::InputType(static_cast<int>(args));
            } else if constexpr (std::is_same_v<T, IR::NNNode::PadCalculation>) {
                nnir_type = nn_ir::PadCalcType(static_cast<int>(args));
            } else if constexpr (std::is_same_v<T, IR::NNNode::PoolType>) {
                nnir_type = nn_ir::PoolType(static_cast<int>(args));
            } else if constexpr (std::is_same_v<T, IR::NNNode::EltwiseType>) {
                nnir_type = nn_ir::EltwiseType(static_cast<int>(args));
            } else if constexpr (std::is_same_v<T, IR::NNNode::ActivationType>) {
                nnir_type = nn_ir::ActivationType(static_cast<int>(args));
            } else if constexpr (std::is_same_v<T, IR::Type::PriorboxType>) {
                nnir_type = nn_ir::PriorboxType(static_cast<int>(args));
            } else if constexpr (std::is_same_v<T, IR::GlobalNode::PartitionModeType>) {
                nnir_type = nn_ir::PartitionMode(static_cast<int>(args));
            } else if constexpr (std::is_same_v<T, IR::GlobalNode::GlobalConcatAxis>) {
                nnir_type = nn_ir::GlobalConcatAxis(static_cast<int>(args));
            } else if constexpr (std::is_same_v<T, IR::GlobalNode::GlobalConcatType>) {
                nnir_type = nn_ir::GlobalConcatType(static_cast<int>(args));
            } else if constexpr (std::is_same_v<T, IR::GlobalNode::SigType>) {
                nnir_type = nn_ir::SigType(static_cast<int>(args));
            } else if constexpr (std::is_same_v<T, IR::GlobalNode::SyncType>) {
                nnir_type = nn_ir::SyncType(static_cast<int>(args));
            } else if constexpr (std::is_same_v<T, IR::Type::TilingSchemeType>) {
                // FIXME : nn_ir and ir not aligned
                std::map<IR::Type::TilingSchemeType, nn_ir::TilingScheme> tiling_scheme_map = {
                    {IR::Type::TilingSchemeType_IFM, nn_ir::TilingScheme::IFM},
                    {IR::Type::TilingSchemeType_WEIGHT, nn_ir::TilingScheme::WEIGHT},
                    {IR::Type::TilingSchemeType_IFM_WEIGHT, nn_ir::TilingScheme::IFM_WEIGHT}};
                nnir_type = tiling_scheme_map[args];
            } else if constexpr (std::is_same_v<T, IR::Type::TilingDirectionType>) {
                nnir_type = nn_ir::Axis(static_cast<int>(args));
            } else if constexpr (std::is_same_v<T, IR::Type::BlobType>) {
                nnir_type = nn_ir::BlobType(static_cast<int>(args));
            } else if constexpr (std::is_same_v<T, IR::Type::QuantType>) {
                nnir_type = nn_ir::QuantType(static_cast<int>(args));
            } else if constexpr (std::is_same_v<T, IR::Type::ShapeType>) {
                nnir_type = nn_ir::ShapeType(static_cast<int>(args));
            } else if constexpr (std::is_same_v<T, IR::Type::DataType>) {
                // FIXME : nn_ir and ir not aligned and need more type
                // Removed with https://github.sec.samsung.net/SAIT-NPU-Compiler/npu_ir/pull/88
                std::map<IR::Type::DataType, nn_ir::DataType> data_type_map = {
                    {IR::Type::DataType_FP_32, nn_ir::DataType::FLOAT32},
                    {IR::Type::DataType_FP_16, nn_ir::DataType::FLOAT16},
                    {IR::Type::DataType_FIXED_64, nn_ir::DataType::INT64},
                    {IR::Type::DataType_FIXED_32, nn_ir::DataType::INT32},
                    {IR::Type::DataType_FIXED_32U, nn_ir::DataType::UINT32},
                    {IR::Type::DataType_FIXED_16, nn_ir::DataType::INT16},
                    {IR::Type::DataType_FIXED_16U, nn_ir::DataType::UINT16},
                    {IR::Type::DataType_FIXED_8, nn_ir::DataType::INT8},
                    {IR::Type::DataType_FIXED_8U, nn_ir::DataType::UINT8},
                    {IR::Type::DataType_FIXED_4, nn_ir::DataType::INT4},
                    {IR::Type::DataType_FIXED_4U, nn_ir::DataType::UINT4},
                    {IR::Type::DataType_BOOL, nn_ir::DataType::BOOL},
                    {IR::Type::DataType_DEVICE, nn_ir::DataType::DEVICE},
                    {IR::Type::DataType_LIST, nn_ir::DataType::LIST},
                    {IR::Type::DataType_NONE, nn_ir::DataType::NONE},
                    {IR::Type::DataType_STRING, nn_ir::DataType::STRING},
                    {IR::Type::DataType_TENSOR, nn_ir::DataType::TENSOR},
                };
                nnir_type = data_type_map[args];
            } else if constexpr (std::is_same_v<T, IR::TargetHardware::Type::NodeExecutionType>) {
                nnir_type = nn_ir::NodeExecutionStepType(static_cast<int>(args));
            } else if constexpr (std::is_same_v<T, IR::TargetHardware::Type::EdgeExecutionType>) {
                nnir_type = nn_ir::EdgeExecutionStepType(static_cast<int>(args));
            } else if constexpr (std::is_same_v<T, IR::TargetHardware::Type::MemoryType>) {
                nnir_type = nn_ir::MemoryType(static_cast<int>(args));
            } else if constexpr (std::is_same_v<T, IR::TargetHardware::Type::MemoryDataType>) {
                nnir_type = nn_ir::MemoryDataType(static_cast<int>(args));
            }
            if constexpr (std::is_same_v<T, IR::TargetHardware::Type::PixelByteOrder>) {
                nnir_type = nn_ir::PixelByteOrder(static_cast<int>(args));
            }
        },
        type);
    return nnir_type;
}

IR_Node_Config_Type_ parseConfigType(NNIR_Node_Config_Type_& type) {
    IR_Node_Config_Type_ ir_type;
    std::visit(
        [&ir_type](auto&& args) {
            using T = std::decay_t<decltype(args)>;
            if constexpr (std::is_same_v<T, nn_ir::InputType>) {
                ir_type = IR::NNNode::InputType(static_cast<int>(args));
            } else if constexpr (std::is_same_v<T, nn_ir::PadCalcType>) {
                ir_type = IR::NNNode::PadCalculation(static_cast<int>(args));
            } else if constexpr (std::is_same_v<T, nn_ir::PoolType>) {
                ir_type = IR::NNNode::PoolType(static_cast<int>(args));
            } else if constexpr (std::is_same_v<T, nn_ir::EltwiseType>) {
                ir_type = IR::NNNode::EltwiseType(static_cast<int>(args));
            } else if constexpr (std::is_same_v<T, nn_ir::ActivationType>) {
                ir_type = IR::NNNode::ActivationType(static_cast<int>(args));
            } else if constexpr (std::is_same_v<T, nn_ir::PriorboxType>) {
                ir_type = IR::Type::PriorboxType(static_cast<int>(args));
            } else if constexpr (std::is_same_v<T, nn_ir::PartitionMode>) {
                ir_type = IR::GlobalNode::PartitionModeType(static_cast<int>(args));
            } else if constexpr (std::is_same_v<T, nn_ir::GlobalConcatAxis>) {
                ir_type = IR::GlobalNode::GlobalConcatAxis(static_cast<int>(args));
            } else if constexpr (std::is_same_v<T, nn_ir::GlobalConcatType>) {
                ir_type = IR::GlobalNode::GlobalConcatType(static_cast<int>(args));
            } else if constexpr (std::is_same_v<T, nn_ir::SigType>) {
                ir_type = IR::GlobalNode::SigType(static_cast<int>(args));
            } else if constexpr (std::is_same_v<T, nn_ir::SyncType>) {
                ir_type = IR::GlobalNode::SyncType(static_cast<int>(args));
            } else if constexpr (std::is_same_v<T, nn_ir::TilingScheme>) {
                std::map<nn_ir::TilingScheme, IR::Type::TilingSchemeType> tiling_scheme_map = {
                    {nn_ir::TilingScheme::IFM, IR::Type::TilingSchemeType_IFM},
                    {nn_ir::TilingScheme::WEIGHT, IR::Type::TilingSchemeType_WEIGHT},
                    {nn_ir::TilingScheme::IFM_WEIGHT, IR::Type::TilingSchemeType_IFM_WEIGHT}};
                ir_type = tiling_scheme_map[args];
            } else if constexpr (std::is_same_v<T, nn_ir::Axis>) {
                ir_type = IR::Type::TilingDirectionType(static_cast<int>(args));
            } else if constexpr (std::is_same_v<T, nn_ir::BlobType>) {
                ir_type = IR::Type::BlobType(static_cast<int>(args));
            } else if constexpr (std::is_same_v<T, nn_ir::QuantType>) {
                ir_type = IR::Type::QuantType(static_cast<int>(args));
            } else if constexpr (std::is_same_v<T, nn_ir::ShapeType>) {
                ir_type = IR::Type::ShapeType(static_cast<int>(args));
            } else if constexpr (std::is_same_v<T, nn_ir::DataType>) {
                std::map<nn_ir::DataType, IR::Type::DataType> data_type_map = {
                    {nn_ir::DataType::FLOAT32, IR::Type::DataType_FP_32},
                    {nn_ir::DataType::FLOAT16, IR::Type::DataType_FP_16},
                    {nn_ir::DataType::INT64, IR::Type::DataType_FIXED_64},
                    {nn_ir::DataType::INT32, IR::Type::DataType_FIXED_32},
                    {nn_ir::DataType::UINT32, IR::Type::DataType_FIXED_32U},
                    {nn_ir::DataType::INT16, IR::Type::DataType_FIXED_16},
                    {nn_ir::DataType::UINT16, IR::Type::DataType_FIXED_16U},
                    {nn_ir::DataType::INT8, IR::Type::DataType_FIXED_8},
                    {nn_ir::DataType::UINT8, IR::Type::DataType_FIXED_8U},
                    {nn_ir::DataType::BOOL, IR::Type::DataType_BOOL},
                    {nn_ir::DataType::DEVICE, IR::Type::DataType_DEVICE},
                    {nn_ir::DataType::LIST, IR::Type::DataType_LIST},
                    {nn_ir::DataType::NONE, IR::Type::DataType_NONE},
                    {nn_ir::DataType::STRING, IR::Type::DataType_STRING},
                    {nn_ir::DataType::TENSOR, IR::Type::DataType_TENSOR},
                };
                ir_type = data_type_map[args];
            } else if constexpr (std::is_same_v<T, nn_ir::NodeExecutionStepType>) {
                ir_type = IR::TargetHardware::Type::NodeExecutionType(static_cast<int>(args));
            } else if constexpr (std::is_same_v<T, nn_ir::EdgeExecutionStepType>) {
                ir_type = IR::TargetHardware::Type::EdgeExecutionType(static_cast<int>(args));
            } else if constexpr (std::is_same_v<T, nn_ir::MemoryType>) {
                ir_type = IR::TargetHardware::Type::MemoryType(static_cast<int>(args));
            } else if constexpr (std::is_same_v<T, nn_ir::MemoryDataType>) {
                ir_type = IR::TargetHardware::Type::MemoryDataType(static_cast<int>(args));
            }
            if constexpr (std::is_same_v<T, nn_ir::PixelByteOrder>) {
                ir_type = IR::TargetHardware::Type::PixelByteOrder(static_cast<int>(args));
            }
        },
        type);
    return ir_type;
}
} // namespace nn_ir
} // namespace nn_compiler
