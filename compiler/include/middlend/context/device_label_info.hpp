#pragma once

#include "compiler/include/middlend/common/log.hpp"
#include "ir/include/nn_ir.hpp"

#include <vector>

namespace nn_compiler {

///@brief Maintain a map of information about Op labels which indicate target platform(CPU/GPU/PIM)
class DeviceLabelInfo {
 public:
    enum class DeviceType {
        CPU = 0,
        GPU,
        PIM
    };

 public:
    DeviceLabelInfo() { device_label_map_ = std::make_shared<std::map<std::string, DeviceType> >(); }

    DeviceLabelInfo(const DeviceLabelInfo& other) = default;

    DeviceLabelInfo& operator=(const DeviceLabelInfo& other) = default;

    void addOpDeviceLabel(const std::string& op_name, const DeviceType& device_label);

    std::shared_ptr<std::map<std::string, DeviceType> >& getOpDeviceLabel() { return device_label_map_; };

    void changeOpDeviceLabel(const std::string& op_name, const DeviceType& device_label);

    int getCPUOpNumber() { return CPU_Op_number_; }

    int getGPUOpNumber() { return GPU_Op_number_; }

    int getPIMOpNumber() { return PIM_Op_number_; }

 private:
    std::shared_ptr<std::map<std::string, DeviceType> > device_label_map_ = nullptr;

    int CPU_Op_number_ = 0;
    int GPU_Op_number_ = 0;
    int PIM_Op_number_ = 0;

    void updateDeviceOpNumber(const DeviceType& device_label);
};

} // namespace nn_compiler
