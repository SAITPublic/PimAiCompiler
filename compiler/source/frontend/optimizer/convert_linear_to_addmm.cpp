#include "compiler/include/frontend/optimizer/convert_linear_to_addmm.h"
#include "ir/include/utils/graph_util.h"

namespace nn_compiler
{
namespace frontend
{
ConvertLinearToAddmm::ConvertLinearToAddmm() {}

bool ConvertLinearToAddmm::fitCondition(std::unique_ptr<nn_compiler::ir::NNModel>& model)
{
    auto graphs = model->getGraphs();
    for (auto graph : graphs) {
        for (auto layer : graph->getLayers()) {
            if (layer->getType() == nn_compiler::ir::LayerType::ATENLINEAR) {
                linear_layers_.push_back(layer);
            }
        }
    }

    return (linear_layers_.size() != 0);
}

void ConvertLinearToAddmm::run(std::unique_ptr<nn_compiler::ir::NNModel>& model)
{
    DLOG(INFO) << "ConvertLinearToAddmm::run is called.";

    // there will be only one graph after take_in_body_net pass.
    auto graph = model->getGraphs()[0];

    for (auto layer : linear_layers_) {
        auto in_stensor_ids = layer->getInSTensorID();
        auto out_stensor_id = layer->getOutSTensorID();
        assert(in_stensor_ids.size() == 3);  // input, weight, bias

        auto type = nn_compiler::ir::LayerType::ATENADDMM;
        auto name = "converted_addmm_of_" + layer->getName();
        auto new_addmm_layer = std::make_shared<nn_compiler::ir::AtenAddmmLayer>(name, type);
        // aten::linear(input, weight, bias) -> aten::addmm(bias, input, weight, ,)
        std::vector<uint32_t> in_ids_of_addmm = {in_stensor_ids[2], in_stensor_ids[0], in_stensor_ids[1]};
        new_addmm_layer->setInSTensorID(in_ids_of_addmm);
        new_addmm_layer->setOutSTensorID(out_stensor_id);  // same output id

        auto layer_position = graph->getLayerPos(layer);
        graph->deleteLayer(layer->getID());
        graph->addLayer2pos(new_addmm_layer, layer_position - 1);
    }
}

}  // namespace frontend
}  // namespace nn_compiler
