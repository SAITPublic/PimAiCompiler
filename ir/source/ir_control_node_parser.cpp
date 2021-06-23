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
    Log::IR::E_IF(constant_node == nullptr) << "IRCONTROLNodeParser::parseControlNode<Control::PrimConstantNode>() => wrong node type!";

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

template <>
std::unique_ptr<nn_ir::CONTROLNode>
IRCONTROLNodeParser::parseControlNode<IR::CONTROLNode::AnyType_PrimDeviceNode>(const IR::ControlNode*    ir_node,
                                                           const nn_ir::NodeInfo& node_info) {
    auto constant_node = ir_node->control_node_as_PrimDeviceNode();
    Log::IR::E_IF(constant_node == nullptr)
    << "IRCONTROLNodeParser::parseNNNode<Control::CONTROLNode>() => wrong node type!";

    return std::make_unique<nn_ir::PrimDeviceNode>(node_info);
}

template <>
std::unique_ptr<nn_ir::CONTROLNode>
IRCONTROLNodeParser::parseControlNode<IR::CONTROLNode::AnyType_PrimDtypeNode>(const IR::ControlNode*    ir_node,
                                                           const nn_ir::NodeInfo& node_info) {
    auto constant_node = ir_node->control_node_as_PrimDtypeNode();
    Log::IR::E_IF(constant_node == nullptr)
    << "IRCONTROLNodeParser::parseNNNode<Control::CONTROLNode>() => wrong node type!";

    return std::make_unique<nn_ir::PrimDtypeNode>(node_info);
}

template <>
std::unique_ptr<nn_ir::CONTROLNode>
IRCONTROLNodeParser::parseControlNode<IR::CONTROLNode::AnyType_PrimIfNode>(const IR::ControlNode*    ir_node,
                                                           const nn_ir::NodeInfo& node_info) {
    auto constant_node = ir_node->control_node_as_PrimIfNode();
    Log::IR::E_IF(constant_node == nullptr)
        << "IRCONTROLNodeParser::parseNNNode<Control::CONTROLNode>() => wrong node type!";

    return std::make_unique<nn_ir::PrimIfNode>(node_info);
}

template <>
std::unique_ptr<nn_ir::CONTROLNode>
IRCONTROLNodeParser::parseControlNode<IR::CONTROLNode::AnyType_PrimListConstructNode>(const IR::ControlNode*    ir_node,
                                                           const nn_ir::NodeInfo& node_info) {
    auto constant_node = ir_node->control_node_as_PrimListConstructNode();
    Log::IR::E_IF(constant_node == nullptr)
    << "IRCONTROLNodeParser::parseNNNode<Control::CONTROLNode>() => wrong node type!";

    return std::make_unique<nn_ir::PrimListConstructNode>(node_info);
}

template <>
std::unique_ptr<nn_ir::CONTROLNode>
IRCONTROLNodeParser::parseControlNode<IR::CONTROLNode::AnyType_PrimLoopNode>(const IR::ControlNode*    ir_node,
                                                           const nn_ir::NodeInfo& node_info) {
    auto constant_node = ir_node->control_node_as_PrimLoopNode();
    Log::IR::E_IF(constant_node == nullptr)
        << "IRCONTROLNodeParser::parseNNNode<Control::CONTROLNode>() => wrong node type!";
    // FIXME: Check if none?
    auto trip_count = constant_node->trip_count();
    auto cond = constant_node->cond();

    return std::make_unique<nn_ir::PrimLoopNode>(node_info, trip_count, cond);
}

template <>
std::unique_ptr<nn_ir::CONTROLNode>
IRCONTROLNodeParser::parseControlNode<IR::CONTROLNode::AnyType_PrimTupleConstructNode>(const IR::ControlNode*      ir_node,
                                                                                       const nn_ir::NodeInfo& node_info) {
    auto tuple_construct_node = ir_node->control_node_as_PrimTupleConstructNode();
    Log::IR::E_IF(tuple_construct_node == nullptr)
    << "IRCONTROLNodeParser::parseControlNode<Control::PrimTupleConstructNode>() => wrong node type!";

    return std::make_unique<nn_ir::PrimTupleConstructNode>(node_info);
}

} // namespace nn_compiler
