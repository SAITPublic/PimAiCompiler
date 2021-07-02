#pragma once

#include "compiler/include/common/log.hpp"
#include "ir/include/nn_ir.hpp"
#include "compiler/include/middlend/context/device_label_info.hpp"
#include <vector>

namespace nn_compiler {

class MemoryLabelInfo {
 public:
    MemoryLabelInfo() { memory_label_map_ = std::make_shared<std::map<EDGE_ID_T, DeviceLabelInfo::DeviceType> >(); }

    MemoryLabelInfo(const MemoryLabelInfo& other) = default;

    MemoryLabelInfo& operator=(const MemoryLabelInfo& other) = default;

    void addEdgeMemoryLabel(const EDGE_ID_T edge_id, const DeviceLabelInfo::DeviceType& memory_label);

    void changeEdgeMemoryLabel(const EDGE_ID_T edge_id, const DeviceLabelInfo::DeviceType& memory_label);

 private:
    std::shared_ptr<std::map<EDGE_ID_T, DeviceLabelInfo::DeviceType> > memory_label_map_ = nullptr;
};

} // namespace nn_compiler
