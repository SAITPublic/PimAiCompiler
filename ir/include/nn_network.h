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

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "ir/include/layers/nn_layer.h"
#include "ir/include/tensors/data_tensor.h"
#include "ir/include/tensors/shape_tensor.h"

namespace nn_compiler {
namespace ir {

class NNNetwork {
 public:
    NNNetwork() {
    }

    explicit NNNetwork(const std::shared_ptr<NNNetwork>& network) {
        name_ = network->getName();
        layers_ = network->getLayers();
    }

    void addLayer(const std::shared_ptr<NNLayer> layer) {
        layers_.push_back(std::move(layer));
    }

    void addLayer2pos(const std::shared_ptr<NNLayer> layer, uint32_t pos) {
        if (pos != layers_.size() - 1) {
            layers_.insert(layers_.begin() + pos + 1, layer);
        } else {
            layers_.push_back(layer);
        }
    }

    int32_t getLayerPos(const std::shared_ptr<NNLayer> layer) {
        int32_t ret = -1;
        for (auto pos = 0; pos < layers_.size(); pos++) {
            if (layer->getID() == layers_[pos]->getID()) {
                ret = pos;
                break;
            }
        }
        return ret;
    }

    std::shared_ptr<NNLayer> getLayerByID(uint32_t id) {
        for (auto layer : layers_) {
            if (layer->getID() == id) {
                return layer;
            }
        }
        return nullptr;
    }

    std::shared_ptr<NNLayer> getLayerByPosition(uint32_t pos) {
        if (pos < layers_.size()) {
            return layers_[pos];
        }
        return nullptr;
    }

    uint32_t deleteLayer(uint32_t id) {
        uint32_t idx  = 0;
        for (auto iter = layers_.begin(); iter != layers_.end(); iter++) {
            if ((*iter)->getID() == id) {
                idx = std::distance(layers_.begin(), iter);
                layers_.erase(layers_.begin() + idx);
                return idx;
            }
        }
        return idx;
    }

    std::vector<std::shared_ptr<NNLayer>>& getLayers() {
        return layers_;
    }

    void addSTensor(std::pair<uint32_t, std::shared_ptr<STensor>> shape_tensor) {
        shape_tensors_.insert(shape_tensor);
    }

    void deleteSTensor(uint32_t shape_tensor_id) {
        shape_tensors_.erase(shape_tensor_id);
    }

    std::map<uint32_t, std::shared_ptr<STensor>> getSTensors() {
        return shape_tensors_;
    }

    void addDTensor(std::pair<uint32_t, std::shared_ptr<DTensor>> data_tensor) {
        data_tensors_.insert(std::move(data_tensor));
    }

    std::map<uint32_t, std::shared_ptr<DTensor>> getDTensors() {
        return data_tensors_;
    }

    void addGraphInTensorID(uint32_t id) {
        in_tensor_ids_.push_back(id);
    }

    std::vector<uint32_t> getGraphInTensorID() {
        return in_tensor_ids_;
    }

    void addGraphOutTensorID(uint32_t id) {
        out_tensor_ids_.push_back(id);
    }

    void renewGraphOutTensorID(uint32_t idx, uint32_t id) {
        CHECK_LT(idx, out_tensor_ids_.size());
        out_tensor_ids_[idx] = id;
    }

    std::vector<uint32_t> getGraphOutTensorID() {
        return out_tensor_ids_;
    }

    const std::string &getName() { return name_; }

    void setName(const std::string &name) { name_ = name; }

    void printNetwork() {
        DLOG(INFO) << "NNLayer Cnt: " << layers_.size();
        for (size_t i = 0; i < layers_.size(); i++) {
            layers_[i]->printInfo();
        }
    }

    ~NNNetwork() = default;

 private:
    std::string name_;

    std::vector<uint32_t> in_tensor_ids_;

    std::vector<uint32_t> out_tensor_ids_;

    std::vector<std::shared_ptr<NNLayer>> layers_;

    std::map<uint32_t, std::shared_ptr<STensor>> shape_tensors_;

    std::map<uint32_t, std::shared_ptr<DTensor>> data_tensors_;
};

}  // namespace ir
}  // namespace nn_compiler
