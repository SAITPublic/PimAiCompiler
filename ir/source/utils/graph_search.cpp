/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system, or translated into any human or computer language in any form by any
 * means, electronic, mechanical, manual or otherwise, or disclosed to third parties without the express written
 * permission of Samsung Electronics. (Use of the Software is restricted to non-commercial, personal or academic,
 * research purpose only)
 */

#include "ir/include/utils/graph_search.h"

namespace nn_compiler
{
namespace ir
{
namespace utils
{
std::vector<std::shared_ptr<ir::NNLayer>> searchPredecessor(const std::shared_ptr<ir::NNLayer> layer,
                                                            const std::unique_ptr<ir::NNModel> &nn_model)
{
    std::vector<std::shared_ptr<ir::NNLayer>> vec;
    auto in_stensor_ids = layer->getInSTensorID();
    auto relation_layers_map = nn_model->getLayerRelationShips();

    for (auto in_id : in_stensor_ids) {
        auto relation_layers = relation_layers_map[in_id];
        for (auto target_layer : relation_layers) {
            for (auto r_layer_out_id : target_layer->getOutSTensorID()) {
                if (r_layer_out_id == in_id) {
                    vec.push_back(target_layer);
                    break;
                }
            }
        }
    }

    return vec;
}

std::vector<std::shared_ptr<ir::NNLayer>> searchSuccessorLayers(const std::shared_ptr<ir::NNLayer> layer,
                                                                const std::unique_ptr<ir::NNModel> &nn_model)
{
    std::vector<std::shared_ptr<ir::NNLayer>> vec;
    auto out_stensor_ids = layer->getOutSTensorID();
    auto relation_layers_map = nn_model->getLayerRelationShips();

    for (auto out_id : out_stensor_ids) {
        auto relation_layers = relation_layers_map[out_id];
        for (auto target_layer : relation_layers) {
            for (auto in_id_ : target_layer->getInSTensorID()) {
                if (in_id_ == out_id) {
                    vec.push_back(target_layer);
                    break;
                }
            }
        }
    }

    return vec;
}

std::map<std::shared_ptr<ir::NNLayer>, uint32_t> searchMapSuccessor(const std::shared_ptr<ir::NNLayer> layer,
                                                                    const std::unique_ptr<ir::NNModel> &nn_model)
{
    std::map<std::shared_ptr<ir::NNLayer>, uint32_t> ret;
    auto out_stensor_ids = layer->getOutSTensorID();
    auto relation_layers_map = nn_model->getLayerRelationShips();

    for (auto out_id : out_stensor_ids) {
        auto relation_layers = relation_layers_map[out_id];
        for (auto target_layer : relation_layers) {
            auto in_stensor_ids = target_layer->getInSTensorID();
            for (auto in_stensor_id = in_stensor_ids.begin(); in_stensor_id != in_stensor_ids.end(); in_stensor_id++) {
                if (*in_stensor_id == out_id) {
                    auto idx = std::distance(in_stensor_ids.begin(), in_stensor_id);
                    ret.insert(std::make_pair(target_layer, idx));
                }
            }
        }
    }

    return ret;
}

std::map<std::shared_ptr<ir::NNLayer>, std::vector<uint32_t>> searchMapSuccessors(
    const std::shared_ptr<ir::NNLayer> layer, const std::unique_ptr<ir::NNModel> &nn_model)
{
    std::map<std::shared_ptr<ir::NNLayer>, std::vector<uint32_t>> ret;
    auto out_stensor_ids = layer->getOutSTensorID();
    auto relation_layers_map = nn_model->getLayerRelationShips();

    for (auto out_id : out_stensor_ids) {
        auto relation_layers = relation_layers_map[out_id];
        for (auto target_layer : relation_layers) {
            std::vector<uint32_t> idx_vec;
            auto in_stensor_ids = target_layer->getInSTensorID();
            for (auto in_stensor_id = in_stensor_ids.begin(); in_stensor_id != in_stensor_ids.end(); in_stensor_id++) {
                if (*in_stensor_id == out_id) {
                    auto idx = std::distance(in_stensor_ids.begin(), in_stensor_id);
                    idx_vec.push_back(idx);
                }
            }
            if (idx_vec.size() > 0) {
                ret.insert(std::make_pair(target_layer, idx_vec));
            }
        }
    }

    return ret;
}

}  // namespace utils
}  // namespace ir
}  // namespace nn_compiler
