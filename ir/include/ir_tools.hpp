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
 * @file.    ir_importer.hpp
 * @brief.   This is IRParser class
 * @details. This header defines IRParser class.
 * @version. 0.1.
 */

#pragma once

#include "flatbuffers/flatbuffers.h"
#include "ir/include/generated/ir_generated.h"
#include "ir/include/nn_ir.hpp"

namespace nn_compiler {

// Save some typing
template <typename Tout, typename Tin>
static inline Tout parseParam(const Tin* ir_param) {
    return std::get<Tout>(nn_ir::parseParam(nn_ir::IR_Node_Param_(ir_param)));
}

template <typename Tout, typename Tin>
static inline Tout parseEnum(Tin ir_type) {
    nn_ir::IR_Node_Config_Type_ ir_variant(ir_type);
    return std::get<Tout>(nn_ir::parseConfigType(ir_variant));
}

static inline nn_ir::ActivationType parseActivation(IR::NNNode::ActivationType ir_type) {
    return parseEnum<nn_ir::ActivationType>(ir_type);
}

static inline nn_ir::DataType convertIrTypeToNNIr(IR::Type::DataType ir_generated) {
    return parseEnum<nn_ir::DataType>(ir_generated);
}

std::unique_ptr<nn_ir::ShiftNode> parseShiftNode(const nn_ir::NodeInfo&       shift_node_info,
                                                 const IR::OPNode::ShiftNode* ir_shift_node);
nn_ir::DataLayout                 parseDataLayout(const IR::TargetHardware::Type::DataLayout* ir_layout);
nn_ir::MemoryInfo                 parseMemoryInfo(const IR::TargetHardware::Type::MemoryInfo* ir_mem_info);

template <typename T>
std::vector<T> makeDataArrFromTypedArray(const IR::Type::TypedArray* data_arr, nn_ir::DataType data_type) {
    std::vector<T> data_vector;

    if (data_arr != nullptr) {
        switch (data_type) {
            case nn_ir::DataType::FLOAT32: {
                auto raw_data = data_arr->f32_arr()->data();
                data_vector.assign(raw_data, raw_data + data_arr->f32_arr()->size());
                break;
            }
            case nn_ir::DataType::INT64: {
                auto raw_data = data_arr->i64_arr()->data();
                data_vector.assign(raw_data, raw_data + data_arr->i64_arr()->size());
                break;
            }
            case nn_ir::DataType::INT32: {
                auto raw_data = data_arr->i32_arr()->data();
                data_vector.assign(raw_data, raw_data + data_arr->i32_arr()->size());
                break;
            }
            case nn_ir::DataType::INT16: {
                auto raw_data = data_arr->i16_arr()->data();
                data_vector.assign(raw_data, raw_data + data_arr->i16_arr()->size());
                break;
            }
            case nn_ir::DataType::INT8: {
                auto raw_data = data_arr->i8_arr()->data();
                data_vector.assign(raw_data, raw_data + data_arr->i8_arr()->size());
                break;
            }
            case nn_ir::DataType::UINT8: {
                auto raw_data = data_arr->ui8_arr()->data();
                data_vector.assign(raw_data, raw_data + data_arr->ui8_arr()->size());
                break;
            }
            default: {
                Log::IR::E() << "makeDataArrFromTypedArray() => unknown data type!";
                break;
            }
        }
    }
    return data_vector;
}

template <typename T>
std::vector<T> makeDataArrFromVector(const flatbuffers::Vector<uint8_t>* data_arr) {
    std::vector<T> data_vector;
    if (!data_arr) {
        return data_vector;
    }
    auto raw_data = reinterpret_cast<const T*>(data_arr->data());
    data_vector.assign(raw_data, raw_data + (data_arr->size() / sizeof(T)));
    return data_vector;
}

template <typename T>
std::vector<T> makeDataArrFromVector(const flatbuffers::Vector<int64_t>* data_arr) {
    std::vector<T> data_vector;
    if (!data_arr) {
        return data_vector;
    }
    auto raw_data = reinterpret_cast<const T*>(data_arr->data());
    data_vector.assign(raw_data, raw_data + (data_arr->size() / sizeof(T)));
    return data_vector;
}

template <typename T>
std::vector<T> getShift(const flatbuffers::Vector<T>* ir_shift_value) {
    std::vector<T> shift;
    if (ir_shift_value != nullptr) {
        auto raw_data  = ir_shift_value->data();
        auto data_size = ir_shift_value->size();
        shift.assign(raw_data, raw_data + data_size);
    }
    return shift;
}
} // namespace nn_compiler
