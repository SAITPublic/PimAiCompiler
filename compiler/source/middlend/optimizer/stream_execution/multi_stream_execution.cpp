#include <string>

#include "ir/include/utils/graph_util.h"
#include "middlend/optimizer/stream_execution/multi_stream_execution.h"

namespace nn_compiler
{
namespace middlend
{
MutiStreamExecution::MutiStreamExecution() {}

bool MutiStreamExecution::fitCondition(std::unique_ptr<nn_compiler::ir::NNModel>& model)
{
    auto graphs = model->getGraphs();
    std::vector<nn_compiler::ir::LayerType> search_layers_type;
    // TODO(zhiyu.zhu): there are some hard-coding in this part, how to make it more general?
    for (auto graph : graphs) {
        for (auto layer : graph->getLayers()) {
            if (layer->getType() == nn_compiler::ir::LayerType::PRIMLISTCONSTRUCT) {
                muti_stream_layers_.push_back(layer);
                bool find_layer = false;
                auto predecessors = ir::utils::searchPredecessor(layer, model);
                if (predecessors.size() == 4 && isSameLayerType(predecessors)) {
                    search_layers_type.push_back(predecessors[0]->getType());
                    predecessors = ir::utils::searchPredecessor(predecessors[0], model);
                    if (predecessors.size() <= 1) {
                        muti_stream_layers_.pop_back();
                        continue;
                    }
                    search_layers_type.push_back(predecessors[0]->getType());
                    predecessors = ir::utils::searchPredecessor(predecessors[0], model);
                    search_layers_type.push_back(predecessors[0]->getType());
                    predecessors = ir::utils::searchPredecessor(predecessors[0], model);
                    search_layers_type.push_back(predecessors[1]->getType());
                    predecessors = ir::utils::searchPredecessor(predecessors[1], model);
                    search_layers_type.push_back(predecessors[0]->getType());
                    predecessors = ir::utils::searchPredecessor(predecessors[0], model);
                    search_layers_type.push_back(predecessors[0]->getType());

                    auto successors = ir::utils::searchSuccessorLayers(predecessors[0], model);
                    if (std::find(search_layers_type.begin(), search_layers_type.end(),
                                  nn_compiler::ir::LayerType::ATENMATMUL) != search_layers_type.end() &&
                        successors.size() == 4 && isSameLayerType(successors)) {
                        muti_stream_layers_.push_back(predecessors[0]);
                        find_layer = true;
                    }
                }
                if (!find_layer) {
                    muti_stream_layers_.pop_back();
                }
            }
        }
    }

    return (muti_stream_layers_.size() != 0);
}

void MutiStreamExecution::run(std::unique_ptr<nn_compiler::ir::NNModel>& model)
{
    DLOG(INFO) << "MutiStreamExecution::run is called.";
    auto graph = model->getGraphs()[0];
    auto layers = graph->getLayers();

    for (int idx = 0; idx < layers.size(); idx++) {
        auto layer = layers[idx];
        if (std::find(muti_stream_layers_.begin(), muti_stream_layers_.end(), layer) != muti_stream_layers_.end()) {
            idx = reorganizeLayerOrders(graph, idx);
        }
    }

    layers = graph->getLayers();
    for (int idx = 0; idx < layers.size(); idx++) {
        auto layer = layers[idx];
        if (std::find(muti_stream_layers_.begin(), muti_stream_layers_.end(), layer) != muti_stream_layers_.end() &&
            layer->getType() != ir::LayerType::PRIMLISTCONSTRUCT) {
            insertMultiStreamLayer(graph, idx);
            layers = graph->getLayers();
        }
    }
}

bool MutiStreamExecution::isSameLayerType(std::vector<std::shared_ptr<ir::NNLayer>>& predecessors)
{
    bool ret = false;
    if (predecessors.size() != 0) {
        if (predecessors.size() == 1) return true;
        nn_compiler::ir::LayerType tmp;
        tmp = predecessors[0]->getType();
        for (int i = 1; i < predecessors.size(); ++i) {
            ret = tmp == predecessors[i]->getType() ? true : false;
        }
    }
    return ret;
}

int MutiStreamExecution::reorganizeLayerOrders(std::shared_ptr<ir::NNGraph>& graph, int start_idx)
{
    auto layers = graph->getLayers();
    int idx = start_idx + 1;

    std::vector<ir::LayerType> layer_types;
    layer_types.push_back(layers[idx]->getType());
    assert((idx + 1) < layers.size());
    // Presupposition: no same datatype for Ops in each stream. Remove this condition later.
    for (int i = idx + 1; i < layers.size(); i++) {
        if (layers[i]->getType() == layer_types[0]) {
            break;
        } else {
            layer_types.push_back(layers[i]->getType());
        }
    }

    int end_idx = start_idx + 1;
    for (end_idx; end_idx < layers.size(); end_idx++) {
        if (std::find(muti_stream_layers_.begin(), muti_stream_layers_.end(), layers[end_idx]) !=
            muti_stream_layers_.end()) {
            break;
        }
    }

    int swap_idx = idx + 1;
    for (auto layer_type : layer_types) {
        layers = graph->getLayers();
        for (int i = swap_idx; i < end_idx; i++) {
            if (layers[i]->getType() == layer_type) {
                graph->swapLayerOrder(swap_idx++, i);
            }
        }
    }

    return end_idx;
}

void MutiStreamExecution::insertMultiStreamLayer(std::shared_ptr<ir::NNGraph>& graph, int idx)
{
    auto layers = graph->getLayers();
    std::vector<std::shared_ptr<ir::NNLayer>> matmul_layers;
    bool find_matmul = false;
    for (idx; idx < layers.size(); idx++) {
        while (layers[idx]->getType() == ir::LayerType::ATENMATMUL) {
            matmul_layers.push_back(layers[idx++]);
            find_matmul = true;
        }
        if (find_matmul) {
            break;
        }
    }
    auto new_type = nn_compiler::ir::LayerType::MULTISTREAM;
    auto new_name = "multi_stream_for_" + matmul_layers[0]->getName();
    auto new_layer = std::make_shared<nn_compiler::ir::MultiStreamLayer>(new_name, new_type);
    new_layer->setLayers(matmul_layers);
    new_layer->setLayerNum(matmul_layers.size());
    new_layer->setStreams();
    graph->addLayer2pos(new_layer, idx - matmul_layers.size() - 1);
}

}  // namespace middlend
}  // namespace nn_compiler
