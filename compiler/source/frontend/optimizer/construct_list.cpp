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

namespace nn_compiler
{

namespace frontend
{

ConstructList::ConstructList() {}

bool ConstructList::fitCondition(std::unique_ptr<nn_compiler::ir::NNModel>& model)
{
    /* fit conditions:
     *   List[int]    = prim::ListConstruct(prim::Constant<int>, prim::Constant<int>, ..., prim::Constant<int>)
     *   List[Tensor] = prim::ListConstruct(prim::Constant<Tensor>, prim::Constant<Tensor>,..., prim::Constant<Tensor>)
     *   List[str]    = prim::ListConstruct(prim::Constant<str>, prim::Constant<str>,..., prim::Constant<str>)
     *   ...
     *   the data from prim::Constant will be assembled and set to a new prim::Variable layer.
     *   As prim::Variable can store a vector of DTensor (prim::Constant can only store one DTensor, which could not
     *   tackle with List[Tensor] case).
     */
    auto graphs = model->getGraphs();
    for (auto graph : graphs) {
        for (auto layer : graph->getLayers()) {
            if (layer->getType() == nn_compiler::ir::LayerType::PRIMLISTCONSTRUCT) {
                auto predecessors = ir::searchPredecessor(layer, graph);
                bool all_constant = true;
                for (unsigned int idx = 0; idx < predecessors.size(); idx++) {
                    // An extra connection has been added between prim::If and the first Op of its then/else body.
                    // In this case, the ListConstruct (as the first Op of prim::If's body) also fit the condition.
                    if (idx == predecessors.size() - 1 && predecessors[idx]->getType() == nn_compiler::ir::LayerType::PRIMIF) {
                        continue;
                    }
                    if (predecessors[idx]->getType() != nn_compiler::ir::LayerType::PRIMCONSTANT) {
                        all_constant = false;
                        break;
                    }
                }

                if (all_constant && predecessors.size()) {
                    std::vector<std::shared_ptr<nn_compiler::ir::DTensor>> dtensor_vec;
                    std::string dtensor_type = "";
                    nn_compiler::ir::DataType data_type = nn_compiler::ir::DataType::UNDEFINED;

                    for (unsigned int idx = 0; idx < predecessors.size(); idx++) {
                        if (idx == predecessors.size() - 1 && predecessors[idx]->getType() == nn_compiler::ir::LayerType::PRIMIF) {
                            continue;
                        }

                        auto constant_layer =
                            std::dynamic_pointer_cast<nn_compiler::ir::PrimConstantLayer>(predecessors[idx]);
                        auto dtensor = constant_layer->getAttr();

                        // capability check
                        if (idx == 0) {
                            dtensor_type = constant_layer->getNType();
                            data_type = dtensor->getDataType();
                        } else if (constant_layer->getNType() != dtensor_type) {
                            Log::FE::E() << "DTensors with different types attempt to construct a list.";
                        } else if (dtensor->getDataType() != data_type) {
                            Log::FE::E() << "Data from DTensors with different types attempt to construct a list.";
                        }

                        dtensor_vec.push_back(dtensor);
                    }

                    auto layer_dtensor_pair = std::make_pair(layer, dtensor_vec);
                    process_layer_and_dtensor_.push_back(layer_dtensor_pair);
                }
            }
        }
    }

    return (process_layer_and_dtensor_.size() != 0);
}

void ConstructList::run(std::unique_ptr<nn_compiler::ir::NNModel>& model)
{
    Log::FE::I() << "ConstructList::run is called.";
    auto graph = model->getGraphs()[0];

    for (auto process_layer_and_dtensor : process_layer_and_dtensor_) {
        auto list_construct_layer = process_layer_and_dtensor.first;
        auto dtensor_vec = process_layer_and_dtensor.second;

        auto predecessors = ir::searchPredecessor(list_construct_layer, graph);
        auto successors = ir::searchSuccessor(list_construct_layer, graph);

        // Store info
        auto layer_id = list_construct_layer->getID();
        auto layer_name = "prim::Variable_" + std::to_string(layer_id);
        auto layer_type = nn_compiler::ir::LayerType::PRIMVARIABLE;
        auto ntype = "List";
        auto out_stensors = list_construct_layer->getOutSTensorID();

        // Tag prim::Constant Ops and remove them in following remove_constant pass
        for (unsigned int idx = 0; idx < predecessors.size(); idx++) {
            if (idx == predecessors.size() - 1 && predecessors[idx]->getType() == nn_compiler::ir::LayerType::PRIMIF) {
                continue;
            }

            auto constant_layer = std::dynamic_pointer_cast<nn_compiler::ir::PrimConstantLayer>(predecessors[idx]);
            auto successors_of_constant_layer = ir::searchSuccessor(predecessors[idx], graph);
            if (successors_of_constant_layer.size() == 1) {
                // This constant only used for prim::ListConstruct. Can be removed.
                constant_layer->setToRemove(true);
            }
        }

        // Prepare for:
        //    prim::If -> prim::ListConstruct -> aten::lstm changes to: prim::If -> aten::lstm
        // which_body_net = "" means no connection with prim::If
        std::string which_body_net = "";
        if (predecessors[predecessors.size() - 1]->getType() == nn_compiler::ir::LayerType::PRIMIF) {
            auto if_layer = predecessors[predecessors.size() - 1];
            auto in_stensor_id = list_construct_layer->getInSTensorID()[predecessors.size() - 1];
            auto out_stensor_ids = if_layer->getOutSTensorID();
            for (unsigned int idx = 0; idx < out_stensor_ids.size(); idx++) {
                if (out_stensor_ids[idx] == in_stensor_id) {
                    if (idx == 0) {
                        which_body_net = "then";
                    } else {
                        which_body_net = "else";
                    }
                }
            }
        }

        // Remove prim::ListConstruct
        auto layer_idx_in_graph = graph->deleteLayer(list_construct_layer->getID());

        // As weight has been set for lstm in layer build stage, if list construct only for lstm weight,
        // the created variable layer could be set to_remove = true and get removed later in remove_prim_layers pass.
        bool only_for_lstm_weight = true;
        if ((std::dynamic_pointer_cast<nn_compiler::ir::PrimConstantLayer>(predecessors[0])->getNType()) == "Tensor") {
            for (auto iter = successors.begin(); iter != successors.end(); iter++) {
                if ((iter->first->getType() != nn_compiler::ir::LayerType::ATENLSTM1 &&
                        iter->first->getType() != nn_compiler::ir::LayerType::ATENLSTM2) ||
                    (iter->first->getType() == nn_compiler::ir::LayerType::ATENLSTM1 && iter->second != 2) ||
                    (iter->first->getType() == nn_compiler::ir::LayerType::ATENLSTM2 && iter->second != 3)) {
                    only_for_lstm_weight = false;
                    break;
                }
            }
        } else {
            only_for_lstm_weight = false;
        }

        // Create and insert a new variable layer to the position
        auto new_variable_layer = std::make_shared<nn_compiler::ir::PrimVariableLayer>(layer_name, layer_type);
        new_variable_layer->setIsConstant(true);
        new_variable_layer->setNType(ntype);
        for (auto dtensor : dtensor_vec) {
            new_variable_layer->setAttr(dtensor);
        }
        for (auto out_stensor : out_stensors) {
            new_variable_layer->addOutSTensorID(out_stensor);
        }
        if (which_body_net == "then") {
            auto if_layer = predecessors[predecessors.size() - 1];
            new_variable_layer->addInSTensorID(if_layer->getOutSTensorID()[0]);
        } else if (which_body_net == "else") {
            auto if_layer = predecessors[predecessors.size() - 1];
            new_variable_layer->addInSTensorID(if_layer->getOutSTensorID()[1]);
        }

        if (only_for_lstm_weight) {
            new_variable_layer->setToRemove(true);
        }

        auto g_layer = std::dynamic_pointer_cast<nn_compiler::ir::NNLayer>(new_variable_layer);
        graph->addLayer2pos(g_layer, layer_idx_in_graph - 1);
    }
}

}  // namespace frontend
}  // namespace nn_compiler
