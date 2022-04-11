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
    auto graphs = nn_model->getGraphs();

    auto in_tensor_ids = layer->getInSTensorID();
    for (auto in_tensor_id : in_tensor_ids) {
        for (auto graph : graphs) {
            for (auto target_layer : graph->getLayers()) {
                if (target_layer->getID() == layer->getID()) {
                    continue;
                }
                auto out_tensor_ids = target_layer->getOutSTensorID();
                if (std::find(out_tensor_ids.begin(), out_tensor_ids.end(), in_tensor_id) != out_tensor_ids.end()) {
                    vec.push_back(target_layer);
                }
            }
        }
    }

    return vec;
}

std::vector<std::shared_ptr<ir::NNLayer>> searchPredecessor(const std::shared_ptr<ir::NNLayer> layer,
                                                            const std::shared_ptr<ir::NNGraph> graph)
{
    std::vector<std::shared_ptr<ir::NNLayer>> vec;
    auto in_tensor_ids = layer->getInSTensorID();
    for (auto in_tensor_id : in_tensor_ids) {
        for (auto target_layer : graph->getLayers()) {
            if (target_layer->getID() == layer->getID()) {
                continue;
            }
            auto out_tensor_ids = target_layer->getOutSTensorID();
            if (std::find(out_tensor_ids.begin(), out_tensor_ids.end(), in_tensor_id) != out_tensor_ids.end()) {
                vec.push_back(target_layer);
            }
        }
    }

    return vec;
}

std::vector<std::shared_ptr<ir::NNLayer>> searchSuccessorLayerOnly(const std::shared_ptr<ir::NNLayer> layer,
                                                                   const std::shared_ptr<ir::NNGraph> graph)
{
    std::vector<std::shared_ptr<ir::NNLayer>> ret;

    auto out_tensor_ids = layer->getOutSTensorID();
    for (auto out_tensor_id : out_tensor_ids) {
        for (auto target_layer : graph->getLayers()) {
            if (target_layer->getID() == layer->getID()) {
                continue;
            }
            auto in_tensor_ids = target_layer->getInSTensorID();
            if (std::find(in_tensor_ids.begin(), in_tensor_ids.end(), out_tensor_id) != in_tensor_ids.end()) {
                ret.push_back(target_layer);
            }
        }
    }

    return ret;
}

std::map<std::shared_ptr<ir::NNLayer>, uint32_t> searchSuccessor(const std::shared_ptr<ir::NNLayer> layer,
                                                                 const std::shared_ptr<ir::NNGraph> graph)
{
    std::map<std::shared_ptr<ir::NNLayer>, uint32_t> ret;

    auto out_tensor_ids = layer->getOutSTensorID();
    for (auto out_tensor_id : out_tensor_ids) {
        for (auto target_layer : graph->getLayers()) {
            if (target_layer->getID() == layer->getID()) {
                continue;
            }
            auto in_tensor_ids = target_layer->getInSTensorID();
            for (auto in_tensor_id = in_tensor_ids.begin(); in_tensor_id != in_tensor_ids.end(); in_tensor_id++) {
                if (*in_tensor_id == out_tensor_id) {
                    auto idx = std::distance(in_tensor_ids.begin(), in_tensor_id);
                    ret.insert(std::make_pair(target_layer, idx));
                }
            }
        }
    }

    return ret;
}

std::map<std::shared_ptr<ir::NNLayer>, std::vector<uint32_t>> searchSuccessors(
    const std::shared_ptr<ir::NNLayer> layer, const std::shared_ptr<ir::NNGraph> graph)
{
    std::map<std::shared_ptr<ir::NNLayer>, std::vector<uint32_t>> ret;

    auto out_tensor_ids = layer->getOutSTensorID();
    for (auto out_tensor_id : out_tensor_ids) {
        for (auto target_layer : graph->getLayers()) {
            if (target_layer->getID() == layer->getID()) {
                continue;
            }
            std::vector<uint32_t> idx_vec;
            auto in_tensor_ids = target_layer->getInSTensorID();
            for (auto in_tensor_id = in_tensor_ids.begin(); in_tensor_id != in_tensor_ids.end(); in_tensor_id++) {
                if (*in_tensor_id == out_tensor_id) {
                    auto idx = std::distance(in_tensor_ids.begin(), in_tensor_id);
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

std::shared_ptr<ir::NNLayer> searchLayerByOutID(uint32_t out_id, const std::shared_ptr<ir::NNGraph> graph)
{
    for (auto cur_layer : graph->getLayers()) {
        auto cur_out_ids = cur_layer->getOutSTensorID();
        if (std::count(cur_out_ids.begin(), cur_out_ids.end(), out_id)) {
            return cur_layer;
        }
    }
    return nullptr;
}

}  // namespace utils
}  // namespace ir
}  // namespace nn_compiler
