#include "compiler/include/middlend/utils/op_basic_util.hpp"

#include "compiler/include/middlend/common/log.hpp"

namespace nn_compiler {

bool OpBasicUtil::isAtenOp(const nn_ir::Node& node) const {
    if (auto nn_node = cast_if<nn_ir::NNNode>(node)) {
        if (is_any_of<nn_ir::AtenAddNode,
                nn_ir::AtenAddmmNode,
                nn_ir::AtenAppendNode,
                nn_ir::AtenCatNode,
                nn_ir::AtenCeilNode,
                nn_ir::AtenCopyNode,
                nn_ir::AtenDeriveIndexNode,
                nn_ir::AtenDimNode,
                nn_ir::AtenDivNode,
                nn_ir::AtenDropoutNode,
                nn_ir::AtenEmbeddingNode,
                nn_ir::AtenEqNode,
                nn_ir::AtenExpandNode,
                nn_ir::AtenFormatNode,
                nn_ir::AtenGetItemNode,
                nn_ir::AtenGtNode,
                nn_ir::AtenIntNode,
                nn_ir::AtenIsNode,
                nn_ir::AtenItemNode,
                nn_ir::AtenLenNode,
                nn_ir::AtenListNode,
                nn_ir::AtenLSTMNode,
                nn_ir::AtenLtNode,
                nn_ir::AtenMatmulNode,
                nn_ir::AtenMaxNode,
                nn_ir::AtenNeNode,
                nn_ir::AtenNegNode,
                nn_ir::AtenReluNode,
                nn_ir::AtenSelectNode,
                nn_ir::AtenSizeNode,
                nn_ir::AtenSliceNode,
                nn_ir::AtenSubNode,
                nn_ir::AtenTensorNode,
                nn_ir::AtenTransposeNode,
                nn_ir::AtenToNode,
                nn_ir::AtenUnsqueezeNode,
                nn_ir::AtenZerosLikeNode,
                nn_ir::AtenZerosNode>(nn_node)) {
            return true;
        }
    }
    return false;
}

bool OpBasicUtil::isPrimOp(const nn_ir::Node& node) const {
    if (auto control_node = cast_if<nn_ir::CONTROLNode>(node)) {
        if (is_any_of<nn_ir::PrimBlockNode,
                nn_ir::PrimConstantNode,
                nn_ir::PrimDataNode,
                nn_ir::PrimDeviceNode,
                nn_ir::PrimDtypeNode,
                nn_ir::PrimEndIfNode,
                nn_ir::PrimEndLoopNode,
                nn_ir::PrimIfNode,
                nn_ir::PrimListConstructNode,
                nn_ir::PrimListUnpackNode,
                nn_ir::PrimLoopIndexNode,
                nn_ir::PrimLoopNode,
                nn_ir::PrimRaiseExceptionNode,
                nn_ir::PrimTupleConstructNode,
                nn_ir::PrimTupleIndexNode,
                nn_ir::PrimTupleUnpackNode,
                nn_ir::PrimUncheckedCastNode,
                nn_ir::PrimUninitializedNode>(control_node)) {
            return true;
        }
    }
    return false;
}

std::string OpBasicUtil::getAtenOpName(const nn_ir::Node& node) const {
    if (isAtenOp(node)) {
        return node.getName();
    }
    Log::ME::E() << "getAtenOpName() only runs for Aten Ops.";
    return "";
}

std::string OpBasicUtil::getPrimOpName(const nn_ir::Node& node) const {
    if (isPrimOp(node)) {
        return node.getName();
    }
    Log::ME::E() << "getPrimOpName() only runs for Prim Ops";
    return "";
}

} // namespace nn_compiler