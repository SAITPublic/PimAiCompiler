#include "ir/include/ir_control_node_parser.hpp"
#include "ir/include/ir_nnnode_parser.hpp"
#include "ir/include/common/log.hpp"
#include "ir/include/ir_includes.hpp"
#include "ir/include/ir_tools.hpp"
#include "ir/include/ir_types.hpp"

namespace nn_compiler {

template <>
std::unique_ptr<nn_ir::CONTROLNode>
IRCONTROLNodeParser::parseControlNode<IR::CONTROLNode::AnyType_PrimConstantNode>(const IR::ControlNode*      ir_node,
                                                           const nn_ir::NodeInfo& node_info) {
    auto constant_node = ir_node->control_node_as_PrimConstantNode();
    Log::IR::E_IF(constant_node == nullptr) << "IRCONTROLNodeParser::parseControlNode<Control::CONTROLNode>() => wrong node type!";

    auto data_arr = constant_node->data();
    auto data = makeDataArrFromVector<uint8_t>(data_arr);
    auto bit_width = constant_node->bit_width();
    auto data_type = constant_node->data_type();
    auto ir_shape = constant_node->tensor_shape();
    nn_ir::Shape4D shape;
    // type is NONE
    if (!ir_shape) {
        shape = {0, 0, 0, 0};
    } else {
        shape = std::get<nn_ir::Shape4D>(nn_ir::parseParam(ir_shape));
    }
    
    return std::make_unique<nn_ir::PrimConstantNode>(node_info, data, bit_width, data_type, shape);
}

} // namespace nn_compiler
