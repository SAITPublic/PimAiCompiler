#include "compiler/include/middlend/context/device_label_info.hpp"

namespace nn_compiler {

void DeviceLabelInfo::addOpDeviceLabel(const std::string& op_name, const DeviceType& device_label) {
    if ((*device_label_map_).find(op_name) != (*device_label_map_).end()) {
        if ((*device_label_map_)[op_name] != device_label) {
            Log::ME::D() << "Device label of Op: " << op_name
                         << " has been changed to " << (device_label == DeviceType::CPU ? "CPU" :
                                                        (device_label == DeviceType::GPU ? "GPU" : "PIM"));
            (*device_label_map_)[op_name] = device_label;
            updateDeviceOpNumber(device_label);
        }
    } else {
        (*device_label_map_)[op_name] = device_label;
        updateDeviceOpNumber(device_label);
    }
}

void DeviceLabelInfo::changeOpDeviceLabel(const std::string& op_name, const DeviceType& device_label) {
    if ((*device_label_map_).find(op_name) == (*device_label_map_).end()) {
        Log::ME::E() << "Op: " << op_name << " has not been assigned with a device label.";
    } else {
        (*device_label_map_)[op_name] = device_label;
        updateDeviceOpNumber(device_label);
    }
}

void DeviceLabelInfo::updateDeviceOpNumber(const DeviceType& device_label) {
    if (device_label == DeviceType::CPU) {
        CPU_Op_number_++;
    } else if (device_label == DeviceType::GPU) {
        GPU_Op_number_++;
    } else if (device_label == DeviceType::PIM) {
        PIM_Op_number_++;
    } else {
        Log::ME::E() << "Undefined device type.";
    }
}

} // namespace nn_compiler
