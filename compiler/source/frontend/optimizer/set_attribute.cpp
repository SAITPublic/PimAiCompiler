#include "frontend/optimizer/set_attribute.h"
#include "ir/include/utils/graph_util.h"

namespace nn_compiler
{
namespace frontend
{
bool SetAttribute::fitCondition(std::unique_ptr<nn_compiler::ir::NNModel> &model)
{
    auto graphs = model->getGraphs();
    for (auto graph : graphs) {
        for (auto layer : graph->getLayers()) {
            if (layer->getType() == nn_compiler::ir::LayerType::PRIMCONSTANT) {
                constant_layers_.push_back(layer);
            } else if (layer->getType() == nn_compiler::ir::LayerType::PRIMVARIABLE) {
                auto variable_layer = std::dynamic_pointer_cast<nn_compiler::ir::PrimVariableLayer>(layer);
                if (variable_layer->getSingleDTensor()) {
                    // variable layer with single DTensor
                    variable_layers_.push_back(layer);
                }
            }
        }
    }

    return (constant_layers_.size() != 0 || variable_layers_.size() != 0);
}

void SetAttribute::run(std::unique_ptr<nn_compiler::ir::NNModel> &model)
{
    DLOG(INFO) << "SetAttribute::run is called.";

    for (auto layer : constant_layers_) {
        auto constant_layer = std::dynamic_pointer_cast<nn_compiler::ir::PrimConstantLayer>(layer);
        if (constant_layer->getNType().find("None") != std::string::npos) {
            continue;
        }
        auto constant_value = constant_layer->getAttr();
        bool remove_layer = true;

        doProcess(layer, model, constant_value, remove_layer);

        constant_layer->setToRemove(remove_layer);
    }

    for (auto layer : variable_layers_) {
        auto variable_layer = std::dynamic_pointer_cast<nn_compiler::ir::PrimVariableLayer>(layer);
        auto variable_data = (variable_layer->getAttr())[0];
        bool remove_layer = true;

        doProcess(layer, model, variable_data, remove_layer);

        variable_layer->setToRemove(remove_layer);
    }

    postProcess();
}

void SetAttribute::doProcess(const std::shared_ptr<nn_compiler::ir::NNLayer> &layer,
                             std::unique_ptr<nn_compiler::ir::NNModel> &model,
                             std::shared_ptr<nn_compiler::ir::DTensor> &data, bool &remove_layer)
{
    auto consumers = ir::utils::searchMapSuccessors(layer, model);
    for (auto consumer : consumers) {
        for (auto inID : consumer.second) {
            std::pair<const std::shared_ptr<nn_compiler::ir::NNLayer>, unsigned int> layer_inID(consumer.first, inID);

            if (helper_->putAttribute(convertLayerTypeToString(consumer.first->getType()), layer_inID, data)) {
                if (edge_remove_helper_.find(consumer.first) == edge_remove_helper_.end()) {
                    std::vector<uint32_t> vec_of_index = {inID};
                    edge_remove_helper_[consumer.first] = vec_of_index;
                } else {
                    edge_remove_helper_[consumer.first].push_back(inID);
                }
            } else {
                remove_layer = false;
            }
        }
    }
}

void SetAttribute::postProcess()
{
    for (auto edge_remover : edge_remove_helper_) {
        auto cur_layer = edge_remover.first;
        auto index_to_remove = edge_remover.second;
        // have to remove edges with index in descending order
        std::sort(index_to_remove.rbegin(), index_to_remove.rend());
        for (auto index : index_to_remove) {
            cur_layer->deleteInSTensorID(index);
        }
    }
}

}  // namespace frontend
}  // namespace nn_compiler
