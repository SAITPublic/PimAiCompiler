#include <string>

#include "compiler/include/frontend/optimizer/construct_list.h"

#include "new_ir/include/layers/aten_lstm1_layer.h"
#include "new_ir/include/layers/aten_lstm2_layer.h"
#include "new_ir/include/layers/prim_constant_layer.h"
#include "new_ir/include/layers/prim_variable_layer.h"
#include "new_ir/include/tensors/data_tensor.h"
#include "new_ir/include/utils/graph_util.h"

#include "new_ir/include/layers/nn_layer.h"
#include "new_ir/include/utils/graph_search.h"

#include "compiler/include/common/log.hpp"
#include "new_ir/include/layers/prim_end_if_layer.h"

#include "compiler/include/frontend/optimizer/update_layer_id.h"

namespace nn_compiler
{

namespace frontend
{

UpdateLayerId::UpdateLayerId() {}

bool UpdateLayerId::fitCondition(std::unique_ptr<nn_compiler::ir::NNModel>& model) { return true; }

void UpdateLayerId::run(std::unique_ptr<nn_compiler::ir::NNModel>& model)
{
    uint32_t layer_id = 0;
    auto graphs = model->getGraphs();
    for (auto graph : graphs) {
        for (auto layer : graph->getLayers()) {
            layer->setID(layer_id++);
        }
    }
}

}  // namespace frontend
}  // namespace nn_compiler
