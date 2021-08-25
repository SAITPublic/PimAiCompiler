#pragma once

#include "common/include/common.hpp"
#include "common/include/types.hpp"

#include "ir/include/control_node.hpp"
#include "ir/include/ir_types.hpp"

namespace nn_compiler
{
namespace nn_ir
{
class PrimConstantNode : public NodeMixin<PrimConstantNode, CONTROLNode>
{
   public:
    explicit PrimConstantNode(const NodeInfo &node_info, std::string ntype, std::vector<uint8_t> data,
                              int32_t bit_width, uint8_t data_type, Shape4D shape, Shape4D stride)
        : NodeMixin(node_info, NodeType::PRIMCONSTANT),
          ntype_(ntype),
          data_(data),
          bit_width_(bit_width),
          data_type_(data_type),
          shape_(shape),
          stride_(stride)
    {
    }

    std::string getNodeTypeAsString(void) const override { return "PrimConstant"; }
    void setData(std::vector<uint8_t> data) { data_ = data; }
    void setBitWidth(int32_t bit_width) { bit_width_ = bit_width; }
    void setDataType(uint8_t data_type) { data_type_ = data_type; }
    void setShape(Shape4D shape) { shape_ = shape; }
    void setStride(Shape4D stride) { stride_ = stride; }
    void setNtype(std::string ntype) { ntype_ = ntype; }

    const std::vector<uint8_t> getData() const { return data_; }
    const int32_t getBitWidth() const { return bit_width_; }
    const uint8_t getDataType() const { return data_type_; }
    const Shape4D getShape() const { return shape_; }
    const Shape4D getStride() const { return stride_; }
    const std::string getNtype() const { return ntype_; }

   private:
    std::vector<uint8_t> data_;
    int32_t bit_width_;
    uint8_t data_type_;
    Shape4D shape_;
    Shape4D stride_;
    std::string ntype_;
};  // class PrimConstantNode

}  // namespace nn_ir
}  // namespace nn_compiler
