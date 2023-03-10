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

#include "frontend/optimizer/remake_dtensor_of_prim_variable.h"
#include "ir/include/utils/graph_util.h"

namespace nn_compiler
{
namespace frontend
{
bool RemakeDTensorOfPrimVariable::checkVariableUsage(const std::shared_ptr<nn_compiler::ir::NNLayer>& layer,
                                                     const std::unique_ptr<ir::NNModel>& nn_model,
                                                     const std::shared_ptr<nn_compiler::ir::DTensor>& data)
{
    auto consumers = ir::utils::searchMapSuccessors(layer, nn_model);
    for (auto consumer : consumers) {
        auto cloned_layer_for_check = consumer.first->clone();
        auto cloned_data_for_check = data->clone();
        for (auto inID : consumer.second) {
            std::pair<const std::shared_ptr<nn_compiler::ir::NNLayer>, unsigned int> layer_inID(cloned_layer_for_check,
                                                                                                inID);

            if (!helper_->putAttribute(convertLayerTypeToString(cloned_layer_for_check->getType()), layer_inID,
                                       cloned_data_for_check)) {
                return false;
            }
        }
    }

    return true;
}

bool RemakeDTensorOfPrimVariable::fitCondition(std::unique_ptr<nn_compiler::ir::NNModel>& model)
{
    auto graphs = model->getGraphs();
    for (auto graph : graphs) {
        for (auto layer : graph->getLayers()) {
            if (layer->getType() == nn_compiler::ir::LayerType::PRIMVARIABLE) {
                auto variable_layer = std::dynamic_pointer_cast<nn_compiler::ir::PrimVariableLayer>(layer);
                auto data = variable_layer->getAttr();
                if (data.size() == 0) {
                    continue;
                }

                if (checkVariableUsage(layer, model, data[0]) && ir::utils::isSingleValueType(data[0]->getDataType())) {
                    variable_layers_.push_back(layer);
                }
            }
        }
    }

    return (variable_layers_.size() != 0);
}

void RemakeDTensorOfPrimVariable::run(std::unique_ptr<nn_compiler::ir::NNModel>& model)
{
    DLOG(INFO) << "RemakeDTensorOfPrimVariable::run is called.";

    // there will be only one graph after take_in_body_net pass.
    auto graph = model->getGraphs()[0];

    for (auto layer : variable_layers_) {
        auto variable_layer = std::dynamic_pointer_cast<nn_compiler::ir::PrimVariableLayer>(layer);
        auto variable_data = variable_layer->getAttr();
        auto new_dtensor = std::make_shared<nn_compiler::ir::DTensor>();

        // TODO(SRCX): support other types
        if (variable_data.size() > 0 && variable_data[0]->getDataType() == nn_compiler::ir::DataType::INT64) {
            int64_t int_arr[variable_data.size()];
            for (unsigned int idx = 0; idx < variable_data.size(); idx++) {
                int_arr[idx] = getSingleValue<int64_t>(variable_data[idx]);
            }
            new_dtensor->setData(int_arr, variable_data.size() * sizeof(int64_t));
            std::vector<int32_t> tensor_shape = {0, 0, 0, variable_data.size()};
            new_dtensor->setTensorShape(nn_compiler::ir::STensor(tensor_shape));
            new_dtensor->setDataType(nn_compiler::ir::DataType::INT64);

            variable_layer->clearAttr();
            variable_layer->setAttr(new_dtensor);
            variable_layer->setSingleDTensor(true);
        }
    }
}

}  // namespace frontend
}  // namespace nn_compiler
