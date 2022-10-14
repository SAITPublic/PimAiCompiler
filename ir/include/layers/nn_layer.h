/*
 * Copyright (C) 2020 Samsung Electronics Co. LTD
 *
 * This software is proprietary of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#pragma once

#include "common/include/common.hpp"
#include "glog/logging.h"
#include "ir/include/tensors/shape_tensor.h"

namespace nn_compiler
{
namespace ir
{
class NNLayer
{
   public:
    NNLayer() {}

    NNLayer(std::string name, LayerType type) : name_(name), type_(type)
    {
        static uint32_t increased_cnt = 0;
        id_ = increased_cnt++;
    }

    explicit NNLayer(const NNLayer& nn_layer)
    {
        in_shape_tensors_ = nn_layer.getInSTensorID();
        out_shape_tensors_ = nn_layer.getOutSTensorID();
        data_tensors_ = nn_layer.data_tensors_;

        name_ = nn_layer.getName();
        type_ = nn_layer.getType();
        id_ = nn_layer.getID();

        if (nn_layer.getActivation()) {
            activation_attr_ = nn_layer.getActivation()->clone();
        }
    }

    virtual std::shared_ptr<NNLayer> clone() { return std::shared_ptr<NNLayer>(new NNLayer(*this)); }

    uint32_t getID() const { return id_; }

    void setID(uint32_t id) { id_ = id; }

    void addInSTensorID(uint32_t input_id) { in_shape_tensors_.push_back(input_id); }

    void deleteInSTensorID(uint32_t idx)
    {
        CHECK_LT(idx, in_shape_tensors_.size());
        in_shape_tensors_.erase(in_shape_tensors_.begin() + idx);
    }

    void renewInSTensorID(uint32_t idx, uint32_t id)
    {
        CHECK_LT(idx, in_shape_tensors_.size());
        in_shape_tensors_[idx] = id;
    }

    std::vector<uint32_t> getInSTensorID() const { return in_shape_tensors_; }

    void setInSTensorID(std::vector<uint32_t> in_shape_tensors) { in_shape_tensors_ = in_shape_tensors; }

    void addOutSTensorID(uint32_t output_id) { out_shape_tensors_.push_back(output_id); }

    void renewOutSTensorID(uint32_t idx, uint32_t id)
    {
        CHECK_LT(idx, out_shape_tensors_.size());
        out_shape_tensors_[idx] = id;
    }

    std::vector<uint32_t> getOutSTensorID() const { return out_shape_tensors_; }

    void setOutSTensorID(std::vector<uint32_t> out_shape_tensors) { out_shape_tensors_ = out_shape_tensors; }

    void addDataTensor(uint32_t id) { data_tensors_.push_back(id); }

    void renewDTensor(uint32_t idx, uint32_t id)
    {
        CHECK_LT(idx, data_tensors_.size());
        data_tensors_[idx] = id;
    }

    std::vector<uint32_t> getDTensors() const { return data_tensors_; }

    void setName(const std::string& name) { name_ = name; }

    const std::string& getName() const { return name_; }

    void setType(LayerType type) { type_ = type; }

    const LayerType& getType() const { return type_; }

    void setPreLayerIDs(std::vector<uint32_t> pre_layer_ids) { pre_layer_ids_ = pre_layer_ids; }

    const std::vector<uint32_t>& getPreLayerIDs() const { return pre_layer_ids_; }

    void setNextLayerIDs(std::vector<uint32_t> next_layer_ids) { next_layer_ids_ = next_layer_ids; }

    const std::vector<uint32_t>& getNextLayerIDs() const { return next_layer_ids_; }

    void setActivation(std::shared_ptr<NNLayer> activation_attr) { activation_attr_ = activation_attr; }

    std::shared_ptr<NNLayer> getActivation() const { return activation_attr_; }

    void printInfo()
    {
        DLOG(INFO) << name_;
        DLOG(INFO) << "{";
        DLOG(INFO) << "    Type is      " << convertLayerTypeToString(type_);
        DLOG(INFO) << "input ZP ";
        DLOG(INFO) << "}";
    }

    virtual ~NNLayer() = default;

   protected:
    std::vector<uint32_t> in_shape_tensors_;
    std::vector<uint32_t> out_shape_tensors_;

    std::vector<uint32_t> data_tensors_;

    uint32_t id_;

    std::string name_;

    LayerType type_;

    std::vector<uint32_t> pre_layer_ids_;
    std::vector<uint32_t> next_layer_ids_;

    std::shared_ptr<NNLayer> activation_attr_ = nullptr;
};

}  // namespace ir
}  // namespace nn_compiler
