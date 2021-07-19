#include "compiler/include/middlend/utils/op_basic_util.hpp"

#include "compiler/include/common/log.hpp"

namespace nn_compiler {

bool OpBasicUtil::isAtenOp(const nn_ir::Node& node) const {
    if (auto nn_node = cast_if<nn_ir::NNNode>(node)) {
        if (is_any_of<nn_ir::AtenAddNode,
                nn_ir::AtenAddmmNode,
                nn_ir::AtenAndNode,
                nn_ir::AtenAnyNode,
                nn_ir::AtenAppendNode,
                nn_ir::AtenArangeNode,
                nn_ir::AtenAsTensorNode,
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
                nn_ir::AtenGatherNode,
                nn_ir::AtenGeNode,
                nn_ir::AtenGetItemNode,
                nn_ir::AtenGtNode,
                nn_ir::AtenIndexNode,
                nn_ir::AtenIndexPutNode,
                nn_ir::AtenIndexSelectNode,
                nn_ir::AtenIntNode,
                nn_ir::AtenIsNode,
                nn_ir::AtenItemNode,
                nn_ir::AtenLeakyReluNode,
                nn_ir::AtenLenNode,
                nn_ir::AtenLinearNode,
                nn_ir::AtenListNode,
                nn_ir::AtenLogNode,
                nn_ir::AtenLogSoftmaxNode,
                nn_ir::AtenLSTMNode,
                nn_ir::AtenLtNode,
                nn_ir::AtenMaskedFillNode,
                nn_ir::AtenMaskedSelectNode,
                nn_ir::AtenMatmulNode,
                nn_ir::AtenMaxNode,
                nn_ir::AtenMaxPool2dNode,
                nn_ir::AtenMinNode,
                nn_ir::AtenNeNode,
                nn_ir::AtenNegNode,
                nn_ir::AtenPackPaddedSequenceNode,
                nn_ir::AtenPadPackedSequenceNode,
                nn_ir::AtenPowNode,
                nn_ir::AtenReluNode,
                nn_ir::AtenSelectNode,
                nn_ir::AtenSetItemNode,
                nn_ir::AtenSizeNode,
                nn_ir::AtenSliceNode,
                nn_ir::AtenSoftmaxNode,
                nn_ir::AtenSqueezeNode,
                nn_ir::AtenSubNode,
                nn_ir::AtenSumNode,
                nn_ir::AtenTanhNode,
                nn_ir::AtenTensorNode,
                nn_ir::AtenTransposeNode,
                nn_ir::AtenToNode,
                nn_ir::AtenTopkNode,
                nn_ir::AtenUnsqueezeNode,
                nn_ir::AtenViewNode,
                nn_ir::AtenWarnNode,
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
                nn_ir::PrimInputNode,
                nn_ir::PrimListConstructNode,
                nn_ir::PrimListUnpackNode,
                nn_ir::PrimLoopIndexNode,
                nn_ir::PrimLoopNode,
                nn_ir::PrimOutputNode,
                nn_ir::PrimRaiseExceptionNode,
                nn_ir::PrimTupleConstructNode,
                nn_ir::PrimTupleIndexNode,
                nn_ir::PrimTupleUnpackNode,
                nn_ir::PrimUncheckedCastNode,
                nn_ir::PrimVariableNode,
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
