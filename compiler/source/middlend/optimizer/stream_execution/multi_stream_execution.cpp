#include <string>

#include "middlend/optimizer/stream_execution/multi_stream_execution.h"
#include "ir/include/utils/graph_util.h"

namespace nn_compiler
{
namespace middlend
{
MutiStreamExecution::MutiStreamExecution() {}


bool MutiStreamExecution::fitCondition(std::unique_ptr<nn_compiler::ir::NNModel>& model)
{
    auto graphs = model->getGraphs();
    std::vector<nn_compiler::ir::LayerType> search_layers_type;
    for (auto graph : graphs) {
        for (auto layer : graph->getLayers()) {
            if (layer->getType() == nn_compiler::ir::LayerType::PRIMLISTCONSTRUCT){
                muti_stream_layers_.push_back(layer);

                auto predecessors = ir::utils::searchPredecessor(layer, model);
                if(predecessors.size() == 4 && isSameLayerType(predecessors)){

                    search_layers_type.push_back(predecessors[0]->getType());
                    auto predecessorslayers = ir::utils::searchPredecessor(predecessors[0], model);
                    search_layers_type.push_back(predecessorslayers[0]->getType());
                    predecessorslayers = ir::utils::searchPredecessor(predecessorslayers[0], model);
                    search_layers_type.push_back(predecessorslayers[0]->getType());
                    predecessorslayers = ir::utils::searchPredecessor(predecessorslayers[0], model);
                    search_layers_type.push_back(predecessorslayers[1]->getType());
                    predecessorslayers = ir::utils::searchPredecessor(predecessorslayers[1], model);
                    search_layers_type.push_back(predecessorslayers[0]->getType());
                    predecessorslayers = ir::utils::searchPredecessor(predecessorslayers[0], model);
                    search_layers_type.push_back(predecessorslayers[0]->getType());

                    auto SuccessorLayers = ir::utils::searchSuccessorLayers(predecessorslayers[0], model);
                    if(std::find(search_layers_type.begin(), search_layers_type.end(), nn_compiler::ir::LayerType::ATENMATMUL) != search_layers_type.end() 
                    && SuccessorLayers.size() == 4 && isSameLayerType(SuccessorLayers)){
                        muti_stream_layers_.push_back(predecessorslayers[0]);
                    }else{
                        muti_stream_layers_.pop_back();
                    }
                }
            }
        }
    }

    return (muti_stream_layers_.size() != 0);;
}


void MutiStreamExecution::run(std::unique_ptr<nn_compiler::ir::NNModel>& model)
{
    DLOG(INFO) << "MutiStreamExecution::run is called.";

}

bool MutiStreamExecution::isSameLayerType(std::vector<std::shared_ptr<ir::NNLayer>>& predecessors){
    bool ret = false;
    if(predecessors.size() != 0){
        if(predecessors.size() == 1) return true;
        nn_compiler::ir::LayerType tmp;
        tmp = predecessors[0]->getType();
        for(int i = 1; i<predecessors.size(); ++i){
            ret = tmp == predecessors[i]->getType() ? true : false;
        }
    }
    return ret;
}

}  // namespace middlend
}  // namespace nn_compiler
