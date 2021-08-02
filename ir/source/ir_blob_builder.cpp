/*
 * Copyright (C) 2019 Samsung Electronics Co. LTD
 *
 * This software is proprietary of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#include "ir/include/ir_blob_builder.hpp"
#include "flatbuffers/flatbuffers.h"
#include "ir/include/common/log.hpp"
#include "ir/include/ir_includes.hpp"
#include "ir/include/ir_tools.hpp"

namespace nn_compiler {

/**
 * @brief.      getOrCreateBlob
 * @details.    This function retrieves Blob instance corresponding to the IR description
 * @param[in].  blob A blob of flatbuffer
 * @param[out].
 * @returns.    Blob pointer
 */
nn_ir::Blob* IRBlobBuilder::getOrCreateBlob(const IR::Blob* ir_blob, nn_ir::NNIR& graph) {
    // Several instructions may refer to the same blob in extraBlobs list.
    // Avoid recreating already existing blobs, as this would remove previous instance
    // from the master list in graph and invalidate its pointer
    BLOB_ID_T    id = ir_blob->id();
    nn_ir::Blob* blob;

    if (graph.haveBlob(id)) {
        blob = graph.getBlob(id);
    } else {
        std::unique_ptr<nn_ir::Blob> newBlob = createBlob(ir_blob, graph);

        // addBlob will destroy the unique_ptr, so cache the pointer now
        blob = newBlob.get();
        graph.addBlob(std::move(newBlob));
    }

    Log::IR::I() << "IRBlobBuilder::getOrCreateBlob : " << *blob;
    return blob;
}

/**
 * @brief.      createBlobFromIR
 * @details.    This function creates Blob instance from IR
 * @param[in].  blob A blob of flatbuffer
 * @param[out].
 * @returns.    return code
 */
std::unique_ptr<nn_ir::Blob> IRBlobBuilder::createBlob(const IR::Blob* blob, nn_ir::NNIR& graph) {
    std::unique_ptr<nn_ir::Blob> g_blob;

    BLOB_ID_T   id             = blob->id();
    std::string name           = blob->name()->str();
    auto        ir_data_type   = blob->data_type();
    auto        ir_data_arr    = blob->data_arr();
    BIT_WIDTH_T bit_width      = blob->bit_width();
    NODE_ID_T   liveness_start = 0;
    NODE_ID_T   liveness_end   = 0;
    int32_t     zero_point     = blob->zero_point();
    auto        ir_frac_len    = blob->frac_len();

    std::vector<uint8_t> meta_arr;

    if (blob->hw_info()) {
        liveness_start = blob->hw_info()->liveness_start_node();
        liveness_end   = blob->hw_info()->liveness_end_node();
    }

    nn_ir::Shape4D alignment_unit = {{.n = 1, .c = 1, .h = 1, .w = 1}};
    bool           compress       = false;
    uint8_t        num_fragments  = 1;

    graph.setNextBlobId(id);

    nn_ir::IR_Node_Config_Type_ ir_blob_type = blob->type();
    auto                        blob_type    = std::get<nn_ir::BlobType>(nn_ir::parseConfigType(ir_blob_type));

    nn_ir::IR_Node_Config_Type_ ir_quant_type = blob->quant_type();
    auto                        quant_type    = std::get<nn_ir::QuantType>(nn_ir::parseConfigType(ir_quant_type));

    // FIXME: IR_quant_level should be handle with attributes
    // switch (ir_quant_level) {
    //     case IR::Type::QuantLevelType_LAYERWISE:
    //         quant_level = nn_ir::QuantLevelType::LAYERWISE;
    //         break;
    //     case IR::Type::QuantLevelType_CHANNELWISE:
    //         quant_level = nn_ir::QuantLevelType::CHANNELWISE;
    //         break;
    //     default:
    //         LOGE(IR, "IRBlobBuilder::createBlob() => unknown quant_level type!\n");
    // }

    nn_ir::IR_Node_Config_Type_ ir_shape   = blob->shape();
    auto                        shape_type = std::get<nn_ir::ShapeType>(nn_ir::parseConfigType(ir_shape));

    nn_ir::IR_Node_Param_ ir_dim = blob->dim();
    auto                  dim    = std::get<nn_ir::Shape4D>(nn_ir::parseParam(ir_dim));

    if (blob->hw_info()) {
        auto ir_alignment_unit = blob->hw_info()->alignment_unit();
        alignment_unit         = std::get<nn_ir::Shape4D>(nn_ir::parseParam(ir_alignment_unit));
        auto compress_info     = blob->hw_info()->compression_info();
        compress               = static_cast<size_t>(compress_info->compression_type()) > 0;
        num_fragments          = compress_info->num_fragments();
        auto ir_meta_arr       = compress_info->metadata_arr();
        meta_arr               = makeDataArrFromVector<uint8_t>(ir_meta_arr);
    }

    auto ir_type = convertIrTypeToNNIr(ir_data_type);
    switch (ir_type) {
        //case nn_ir::DataType::NONE:
        case nn_ir::DataType::UINT32:
            Log::IR::E() << "IRBlobBuilder::createBlob() => unknown data type!";
            break;
        default: {
        }
    }

    std::vector<FRAC_LENGTH_T> frac_len;
    if (ir_frac_len) {
        const int8_t* raw_data = ir_frac_len->data();
        frac_len.assign(raw_data, raw_data + ir_frac_len->size());
    }

    nn_ir::BlobInfo blob_info{id,
                              name,
                              graph,
                              blob_type,
                              ir_type,
                              quant_type,
                              shape_type,
                              dim,
                              alignment_unit,
                              bit_width,
                              {liveness_start, liveness_end},
                              zero_point,
                              compress,
                              frac_len};

#define DATATYPE(NNIR_TYPE, IR_GENERATED_TYPE, C_TYPE)                                        \
    case IR::Type::DataType::DataType_##IR_GENERATED_TYPE: {                                  \
        if (ir_data_arr != nullptr) {                                                         \
            auto data_arr = makeDataArrFromVector<C_TYPE>(ir_data_arr);                       \
            g_blob        = std::make_unique<nn_ir::DataBlob>(blob_info, data_arr, meta_arr); \
        } else {                                                                              \
            g_blob = std::make_unique<nn_ir::FeaturemapBlob>(blob_info);                      \
        }                                                                                     \
        break;                                                                                \
    }

    switch (ir_data_type) {
        DATATYPE(FLOAT64, FP_64, double)
        DATATYPE(FLOAT32, FP_32, float)
        DATATYPE(FLOAT16, FP_16, float16)
        DATATYPE(INT16, FIXED_16, int16_t)
        DATATYPE(UINT16, FIXED_16U, uint16_t)
        DATATYPE(INT8, FIXED_8, int8_t)
        DATATYPE(UINT8, FIXED_8U, uint8_t)
        DATATYPE(INT32, FIXED_32, int32_t)
        DATATYPE(INT64, FIXED_64, int64_t)
        DATATYPE(INT4, FIXED_4, int4_t)
        DATATYPE(UINT4, FIXED_4U, uint4_t)
        DATATYPE(BOOL, BOOL, int8_t)
        DATATYPE(DEVICEL, DEVICE, int8_t)
        DATATYPE(LIST, LIST, int8_t)
        DATATYPE(NONE, NONE, int8_t)
        DATATYPE(STRING, STRING, int8_t)
        DATATYPE(TENSOR, TENSOR, int8_t)
        default: {
            Log::IR::E() << "IRBlobBuilder::createBlob() => unknown data type!";
        }
    }
#undef DATATYPE

    g_blob->setFLCFragments(num_fragments);
    return g_blob;
}
} // namespace nn_compiler
