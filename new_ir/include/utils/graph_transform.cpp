#include "new_ir/include/utils/graph_search.h"
#include "new_ir/include/utils/graph_transform.h"


namespace nn_compiler {
namespace ir {

void deleteLayer(std::shared_ptr<ir::NNNetwork> graph, std::shared_ptr<ir::NNLayer> layer) {
    auto cur_in_ids = layer->getInSTensorID();
    CHECK_EQ(cur_in_ids.size(), 1) << " not supported case";

    auto cur_out_ids = layer->getOutSTensorID();
    CHECK_EQ(cur_out_ids.size(), 1) << " not supported case";

    auto successors = searchSuccessor(layer, graph);
    for (auto successor : successors) {
        auto successor_layer = successor.first;
        auto successor_index = successor.second;
        successor_layer->renewInSTensorID(successor_index, cur_in_ids[0]);
    }

    graph->deleteSTensor(cur_out_ids[0]);

    graph->deleteLayer(layer->getID());
}

}  // namespace ir
}  // namespace nn_compiler
