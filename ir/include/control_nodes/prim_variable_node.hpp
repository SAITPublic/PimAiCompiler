#pragma once

#include "common/include/common.hpp"
#include "common/include/types.hpp"

#include "ir/include/ir_types.hpp"
#include "ir/include/control_node.hpp"

namespace nn_compiler {
namespace nn_ir
{
class PrimVariableNode : public NodeMixin<PrimVariableNode, CONTROLNode>
{
   public:
    explicit PrimVariableNode(const NodeInfo &node_info, std::vector<uint8_t> data, std::vector<Shape4D> shape,
                              std::vector<Shape4D> strides, std::string data_type,
                              std::vector<std::string> tensor_data_type)
        : NodeMixin(node_info, NodeType::PRIMVARIABLE),
          data_(data),
          shape_(shape),
          strides_(strides),
          data_type_(data_type),
          tensor_data_type_(tensor_data_type)
    {
    }

    std::string getNodeTypeAsString(void) const override { return "PrimVariable"; }
    void setData(std::vector<uint8_t> data) { data_ = data; }
    void setShape(std::vector<Shape4D> shape) { shape_ = shape; }
    void setStrides(std::vector<Shape4D> strides) { strides_ = strides; }
    void setDataType(std::string data_type) { data_type_ = data_type; }
    void setTensorDataType(std::vector<std::string> tensor_data_type) { tensor_data_type_ = tensor_data_type; }

    const std::vector<uint8_t> getData() const { return data_; }
    const std::vector<Shape4D> getShape() const { return shape_; }
    const std::vector<Shape4D> getStrides() const { return strides_; }
    const std::string getDataType() const { return data_type_; }
    const std::vector<std::string> getTensorDataType() const { return tensor_data_type_; }

   private:
    std::vector<uint8_t> data_;
    std::vector<Shape4D> shape_;
    std::vector<Shape4D> strides_;
    std::string data_type_;
    std::vector<std::string> tensor_data_type_;
};  // class PrimVariableNode

}  // namespace nn_ir
}  // namespace nn_compiler
