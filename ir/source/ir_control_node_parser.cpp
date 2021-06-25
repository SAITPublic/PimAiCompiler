#include "ir/include/ir_control_node_parser.hpp"
#include "ir/include/ir_nnnode_parser.hpp"
#include "ir/include/common/log.hpp"
#include "ir/include/ir_includes.hpp"
#include "ir/include/ir_tools.hpp"
#include "ir/include/ir_types.hpp"

namespace nn_compiler {

template <>
std::unique_ptr<nn_ir::CONTROLNode>
IRCONTROLNodeParser::parseControlNode<IR::CONTROLNode::AnyType_PrimBlockNode>(const IR::ControlNode*    ir_node,
                                                           const nn_ir::NodeInfo& node_info) {
    auto constant_node = ir_node->control_node_as_PrimBlockNode();
    Log::IR::E_IF(constant_node == nullptr)
        << "IRCONTROLNodeParser::parseNNNode<Control::CONTROLNode>() => wrong node type!";

    return std::make_unique<nn_ir::PrimBlockNode>(node_info);
}

template <>
std::unique_ptr<nn_ir::CONTROLNode>
IRCONTROLNodeParser::parseControlNode<IR::CONTROLNode::AnyType_PrimConstantNode>(const IR::ControlNode*      ir_node,
                                                                                 const nn_ir::NodeInfo& node_info) {
    auto prim_constant_node = ir_node->control_node_as_PrimConstantNode();
    Log::IR::E_IF(prim_constant_node == nullptr) 
    << "IRCONTROLNodeParser::parseControlNode<Control::PrimConstantNode>() => wrong node type!";

    auto data_arr = prim_constant_node->data();
    auto data = makeDataArrFromVector<uint8_t>(data_arr);
    auto bit_width = prim_constant_node->bit_width();
    auto data_type = prim_constant_node->data_type();
    auto ir_shape = prim_constant_node->tensor_shape();
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
IRCONTROLNodeParser::parseControlNode<IR::CONTROLNode::AnyType_PrimDataNode>(const IR::ControlNode*      ir_node,
                                                           const nn_ir::NodeInfo& node_info) {
    auto constant_node = ir_node->control_node_as_PrimDataNode();
    Log::IR::E_IF(constant_node == nullptr)
    << "IRCONTROLNodeParser::parseControlNode<Control::PrimDataNode>() => wrong node type!";

    return std::make_unique<nn_ir::PrimDataNode>(node_info);
}

template <>
std::unique_ptr<nn_ir::CONTROLNode>
IRCONTROLNodeParser::parseControlNode<IR::CONTROLNode::AnyType_PrimDeviceNode>(const IR::ControlNode*    ir_node,
                                                           const nn_ir::NodeInfo& node_info) {
    auto prim_device_node = ir_node->control_node_as_PrimDeviceNode();
    Log::IR::E_IF(prim_device_node == nullptr)
    << "IRCONTROLNodeParser::parseControlNode<Control::PrimDeviceNode>() => wrong node type!";
    return std::make_unique<nn_ir::PrimDeviceNode>(node_info);
}

template <>
std::unique_ptr<nn_ir::CONTROLNode>
IRCONTROLNodeParser::parseControlNode<IR::CONTROLNode::AnyType_PrimDtypeNode>(const IR::ControlNode*    ir_node,
                                                           const nn_ir::NodeInfo& node_info) {
    auto prim_dtype_node = ir_node->control_node_as_PrimDtypeNode();
    Log::IR::E_IF(prim_dtype_node == nullptr)
    << "IRCONTROLNodeParser::parseControlNode<Control::PrimDtypeNode>() => wrong node type!";

    return std::make_unique<nn_ir::PrimDtypeNode>(node_info);
}

template <>
std::unique_ptr<nn_ir::CONTROLNode>
IRCONTROLNodeParser::parseControlNode<IR::CONTROLNode::AnyType_PrimEndIfNode>(const IR::ControlNode*    ir_node,
                                                           const nn_ir::NodeInfo& node_info) {
    auto prim_end_if_node = ir_node->control_node_as_PrimEndIfNode();
    Log::IR::E_IF(prim_end_if_node == nullptr)
    << "IRCONTROLNodeParser::parseControlNode<Control::PrimEndIfNode>() => wrong node type!";

    return std::make_unique<nn_ir::PrimEndIfNode>(node_info);
}

template <>
std::unique_ptr<nn_ir::CONTROLNode>
IRCONTROLNodeParser::parseControlNode<IR::CONTROLNode::AnyType_PrimEndLoopNode>(const IR::ControlNode* ir_node,
                                                           const nn_ir::NodeInfo& node_info) {
    auto end_loop_node = ir_node->control_node_as_PrimEndLoopNode();
    Log::IR::E_IF(end_loop_node == nullptr)
        << "IRCONTROLNodeParser::parseControlNode<Control::PrimEndLoopNode>() => wrong node type!";

    return std::make_unique<nn_ir::PrimEndLoopNode>(node_info);
}

template <>
std::unique_ptr<nn_ir::CONTROLNode>
IRCONTROLNodeParser::parseControlNode<IR::CONTROLNode::AnyType_PrimIfNode>(const IR::ControlNode*    ir_node,
                                                           const nn_ir::NodeInfo& node_info) {
    auto prim_if_node = ir_node->control_node_as_PrimIfNode();
    Log::IR::E_IF(prim_if_node == nullptr)
        << "IRCONTROLNodeParser::parseControlNode<Control::PrimIfNode>() => wrong node type!";

    return std::make_unique<nn_ir::PrimIfNode>(node_info);
}

template <>
std::unique_ptr<nn_ir::CONTROLNode>
IRCONTROLNodeParser::parseControlNode<IR::CONTROLNode::AnyType_PrimListConstructNode>(const IR::ControlNode*    ir_node,
                                                           const nn_ir::NodeInfo& node_info) {
    auto prim_list_construct_node = ir_node->control_node_as_PrimListConstructNode();
    Log::IR::E_IF(prim_list_construct_node == nullptr)
    << "IRCONTROLNodeParser::parseControlNode<Control::PrimListConstructNode>() => wrong node type!";

    return std::make_unique<nn_ir::PrimListConstructNode>(node_info);
}

template <>
std::unique_ptr<nn_ir::CONTROLNode>
IRCONTROLNodeParser::parseControlNode<IR::CONTROLNode::AnyType_PrimLoopIndexNode>(const IR::ControlNode*    ir_node,
                                                           const nn_ir::NodeInfo& node_info) {
    auto prim_loop_node = ir_node->control_node_as_PrimLoopIndexNode();
    Log::IR::E_IF(prim_loop_node == nullptr)
        << "IRCONTROLNodeParser::parseControlNode<Control::CONTROLNode>() => wrong node type!";

    auto index = prim_loop_node->index();

    return std::make_unique<nn_ir::PrimLoopIndexNode>(node_info, index);
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
IRCONTROLNodeParser::parseControlNode<IR::CONTROLNode::AnyType_PrimRaiseExceptionNode>
                                    (const IR::ControlNode*      ir_node,
                                     const nn_ir::NodeInfo& node_info) {
    auto prim_raise_exception_node = ir_node->control_node_as_PrimRaiseExceptionNode();
    Log::IR::E_IF(prim_raise_exception_node == nullptr)
    << "IRCONTROLNodeParser::parseControlNode<Control::PrimRaiseExceptionNode>() => wrong node type!";

    return std::make_unique<nn_ir::PrimRaiseExceptionNode>(node_info);
}

template <>
std::unique_ptr<nn_ir::CONTROLNode>
IRCONTROLNodeParser::parseControlNode<IR::CONTROLNode::AnyType_PrimTupleConstructNode>(const IR::ControlNode*      ir_node,
                                                                                       const nn_ir::NodeInfo& node_info) {
    auto prim_tuple_construct_node = ir_node->control_node_as_PrimTupleConstructNode();
    Log::IR::E_IF(prim_tuple_construct_node == nullptr) 
    << "IRCONTROLNodeParser::parseControlNode<Control::PrimTupleConstructNode>() => wrong node type!";

    return std::make_unique<nn_ir::PrimTupleConstructNode>(node_info);
}

template <>
std::unique_ptr<nn_ir::CONTROLNode>
IRCONTROLNodeParser::parseControlNode<IR::CONTROLNode::AnyType_PrimTupleIndexNode>(const IR::ControlNode*      ir_node,
                                                                                   const nn_ir::NodeInfo& node_info) {
    auto prim_tuple_index_node = ir_node->control_node_as_PrimTupleIndexNode();
    Log::IR::E_IF(prim_tuple_index_node == nullptr)
    << "IRCONTROLNodeParser::parseControlNode<Control::PrimTupleIndexNode>() => wrong node type!";
    int64_t index = prim_tuple_index_node->index();
    return std::make_unique<nn_ir::PrimTupleIndexNode>(node_info, index);
}

template <>
std::unique_ptr<nn_ir::CONTROLNode>
IRCONTROLNodeParser::parseControlNode<IR::CONTROLNode::AnyType_PrimTupleUnpackNode>
                                    (const IR::ControlNode*      ir_node,
                                     const nn_ir::NodeInfo& node_info) {
    auto prim_tuple_unpack_node = ir_node->control_node_as_PrimTupleUnpackNode();
    Log::IR::E_IF(prim_tuple_unpack_node == nullptr)
    << "IRCONTROLNodeParser::parseControlNode<Control::PrimTupleUnpackNode>() => wrong node type!";

    return std::make_unique<nn_ir::PrimTupleUnpackNode>(node_info);
}

template <>
std::unique_ptr<nn_ir::CONTROLNode>
IRCONTROLNodeParser::parseControlNode<IR::CONTROLNode::AnyType_PrimUncheckedCastNode>
                                    (const IR::ControlNode*      ir_node,
                                     const nn_ir::NodeInfo& node_info) {
    auto prim_unchecked_cast_node = ir_node->control_node_as_PrimUncheckedCastNode();
    Log::IR::E_IF(prim_unchecked_cast_node == nullptr)
    << "IRCONTROLNodeParser::parseControlNode<Control::PrimUncheckedCastNode>() => wrong node type!";

    return std::make_unique<nn_ir::PrimUncheckedCastNode>(node_info);
}

template <>
std::unique_ptr<nn_ir::CONTROLNode>
IRCONTROLNodeParser::parseControlNode<IR::CONTROLNode::AnyType_PrimUninitializedNode>(const IR::ControlNode*      ir_node,
                                                                                      const nn_ir::NodeInfo& node_info) {
    auto tuple_construct_node = ir_node->control_node_as_PrimUninitializedNode();
    Log::IR::E_IF(tuple_construct_node == nullptr)
    << "IRCONTROLNodeParser::parseControlNode<Control::PrimUninitializedNode>() => wrong node type!";

    return std::make_unique<nn_ir::PrimUninitializedNode>(node_info);
}

} // namespace nn_compiler
