#include "compiler/include/middlend/context/memory_label_info.hpp"

namespace nn_compiler {

void MemoryLabelInfo::addEdgeMemoryLabel(const EDGE_ID_T edge_id, const DeviceLabelInfo::DeviceType& memory_label){
    if ((*memory_label_map_).find(edge_id) != (*memory_label_map_).end()) {
        if ((*memory_label_map_)[edge_id] != memory_label) {
            Log::ME::D() << "Memory label of Edge: " << edge_id
                         << " has been changed to " << "GPU";
            (*memory_label_map_)[edge_id] = memory_label;
        }
    } else {
        (*memory_label_map_)[edge_id] = memory_label;
    }
}

void MemoryLabelInfo::changeEdgeMemoryLabel(const EDGE_ID_T edge_id, const DeviceLabelInfo::DeviceType& memory_label) {
    //TODO
}

} // namespace nn_compiler
