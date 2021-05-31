import IR.NnNode
import IR.NNNode
# Torch Aten Ops
import IR.NNNode.AtenAddNode
import IR.NNNode.AtenAddmmNode
import IR.NNNode.AtenAppendNode
import IR.NNNode.AtenCatNode
import IR.NNNode.AtenCeilNode
import IR.NNNode.AtenCopyNode
import IR.NNNode.AtenDeriveIndexNode
import IR.NNNode.AtenDimNode
import IR.NNNode.AtenDivNode
import IR.NNNode.AtenDropoutNode
import IR.NNNode.AtenEmbeddingNode
import IR.NNNode.AtenEqNode
import IR.NNNode.AtenExpandNode
import IR.NNNode.AtenFormatNode
import IR.NNNode.AtenGetItemNode
import IR.NNNode.AtenGtNode
import IR.NNNode.AtenIntNode
import IR.NNNode.AtenIsNode
import IR.NNNode.AtenItemNode
import IR.NNNode.AtenLSTMNode
import IR.NNNode.AtenLenNode
import IR.NNNode.AtenListNode
import IR.NNNode.AtenLtNode
import IR.NNNode.AtenMatmulNode
import IR.NNNode.AtenMaxNode
import IR.NNNode.AtenNeNode
import IR.NNNode.AtenNegNode
import IR.NNNode.AtenNotNode
import IR.NNNode.AtenReluNode
import IR.NNNode.AtenSelectNode
import IR.NNNode.AtenSizeNode
import IR.NNNode.AtenSliceNode
import IR.NNNode.AtenSubNode
import IR.NNNode.AtenTensorNode
import IR.NNNode.AtenToNode
import IR.NNNode.AtenTransposeNode
import IR.NNNode.AtenUnsqueezeNode
import IR.NNNode.AtenZerosLikeNode
import IR.NNNode.AtenZerosNode

# Torch Prim Ops
import  IR.CONTROLNode
import  IR.CONTROLNode.AnyType
import IR.CONTROLNode.PrimBlockNode
import IR.CONTROLNode.PrimCallMethodNode
import IR.CONTROLNode.PrimConstantNode
import IR.CONTROLNode.PrimDataNode
import IR.CONTROLNode.PrimDeviceNode
import IR.CONTROLNode.PrimDtypeNode
import IR.CONTROLNode.PrimEndIfNode
import IR.CONTROLNode.PrimEndLoopNode
import IR.CONTROLNode.PrimIfNode
import IR.CONTROLNode.PrimListConstructNode
import IR.CONTROLNode.PrimListUnpackNode
import IR.CONTROLNode.PrimLoopIndexNode
import IR.CONTROLNode.PrimLoopNode
import IR.CONTROLNode.PrimRaiseExceptionNode
import IR.CONTROLNode.PrimTupleConstructNode
import IR.CONTROLNode.PrimTupleIndexNode
import IR.CONTROLNode.PrimTupleUnpackNode
import IR.CONTROLNode.PrimUncheckedCastNode
import IR.CONTROLNode.PrimUninitializedNode


'''
Torch Ops can be devided into 2 parts: torch::Aten Ops and torch::Prim Ops
Not all Ops have attributes, the table should be updated when add new torch Ops
'''


def create_aten_ops_map():
    '''
    create a dict that contains all torch::aten Ops of RNNT
    '''
    op_dict = {
        IR.NNNode.AnyType.AnyType().AtenAddNode : 'AtenAddNode',
        IR.NNNode.AnyType.AnyType().AtenAddmmNode : 'AtenAddmmNode',
        IR.NNNode.AnyType.AnyType().AtenAppendNode : 'AtenAppendNode',
        IR.NNNode.AnyType.AnyType().AtenCatNode : 'AtenCatNode',
        IR.NNNode.AnyType.AnyType().AtenCeilNode : 'AtenCeilNode',
        IR.NNNode.AnyType.AnyType().AtenCopyNode : 'AtenCopyNode',
        IR.NNNode.AnyType.AnyType().AtenDeriveIndexNode : 'AtenDeriveIndexNode',
        IR.NNNode.AnyType.AnyType().AtenDimNode : 'AtenDimNode',
        IR.NNNode.AnyType.AnyType().AtenDivNode : 'AtenDivNode',
        IR.NNNode.AnyType.AnyType().AtenDropoutNode : 'AtenDropoutNode',
        IR.NNNode.AnyType.AnyType().AtenEmbeddingNode : 'AtenEmbeddingNode',
        IR.NNNode.AnyType.AnyType().AtenEqNode : 'AtenEqNode',
        IR.NNNode.AnyType.AnyType().AtenExpandNode : 'AtenExpandNode',
        IR.NNNode.AnyType.AnyType().AtenFormatNode : 'AtenFormatNode',
        IR.NNNode.AnyType.AnyType().AtenGetItemNode : 'AtenGetItemNode',
        IR.NNNode.AnyType.AnyType().AtenGtNode : 'AtenGtNode',
        IR.NNNode.AnyType.AnyType().AtenIntNode : 'AtenIntNode',
        IR.NNNode.AnyType.AnyType().AtenIsNode : 'AtenIsNode',
        IR.NNNode.AnyType.AnyType().AtenItemNode : 'AtenItemNode',
        IR.NNNode.AnyType.AnyType().AtenLSTMNode : 'AtenLSTMNode',
        IR.NNNode.AnyType.AnyType().AtenLenNode : 'AtenLenNode',
        IR.NNNode.AnyType.AnyType().AtenListNode : 'AtenListNode',
        IR.NNNode.AnyType.AnyType().AtenLtNode : 'AtenLtNode',
        IR.NNNode.AnyType.AnyType().AtenMatmulNode : 'AtenMatmulNode',
        IR.NNNode.AnyType.AnyType().AtenMaxNode : 'AtenMaxNode',
        IR.NNNode.AnyType.AnyType().AtenNeNode : 'AtenNeNode',
        IR.NNNode.AnyType.AnyType().AtenNegNode : 'AtenNegNode',
        IR.NNNode.AnyType.AnyType().AtenNotNode : 'AtenNotNode',
        IR.NNNode.AnyType.AnyType().AtenReluNode : 'AtenReluNode',
        IR.NNNode.AnyType.AnyType().AtenSelectNode : 'AtenSelectNode',
        IR.NNNode.AnyType.AnyType().AtenSizeNode : 'AtenSizeNode',
        IR.NNNode.AnyType.AnyType().AtenSliceNode : 'AtenSliceNode',
        IR.NNNode.AnyType.AnyType().AtenSubNode : 'AtenSubNode',
        IR.NNNode.AnyType.AnyType().AtenTensorNode : 'AtenTensorNode',
        IR.NNNode.AnyType.AnyType().AtenToNode : 'AtenToNode',
        IR.NNNode.AnyType.AnyType().AtenTransposeNode : 'AtenTransposeNode',
        IR.NNNode.AnyType.AnyType().AtenUnsqueezeNode : 'AtenUnsqueezeNode',
        IR.NNNode.AnyType.AnyType().AtenZerosLikeNode : 'AtenZerosLikeNode',
        IR.NNNode.AnyType.AnyType().AtenZerosNode : 'AtenZerosNode'
    }
    return op_dict


aten_ops_dict = create_aten_ops_map()

'''
these torch::Aten Ops have no attribute
'''
aten_ops_no_attr_dict = {
    'AtenAddNode' : IR.NNNode.AtenAddNode.AtenAddNode(),
    'AtenAddmmNode' : IR.NNNode.AtenAddmmNode.AtenAddmmNode(),
    'AtenAppendNode' : IR.NNNode.AtenAppendNode.AtenAppendNode(),
    'AtenCeilNode' : IR.NNNode.AtenCeilNode.AtenCeilNode(),
    'AtenDimNode' : IR.NNNode.AtenDimNode.AtenDimNode(),
    'AtenDivNode' : IR.NNNode.AtenDivNode.AtenDivNode(),
    'AtenEqNode' : IR.NNNode.AtenEqNode.AtenEqNode(),
    'AtenGetItemNode' : IR.NNNode.AtenGetItemNode.AtenGetItemNode(),
    'AtenGtNode' : IR.NNNode.AtenGtNode.AtenGtNode(),
    'AtenIntNode' : IR.NNNode.AtenIntNode.AtenIntNode(),
    'AtenIsNode' : IR.NNNode.AtenIsNode.AtenIsNode(),
    'AtenItemNode' : IR.NNNode.AtenItemNode.AtenItemNode(),
    'AtenLenNode' : IR.NNNode.AtenLenNode.AtenLenNode(),
    'AtenListNode' : IR.NNNode.AtenListNode.AtenListNode(),
    'AtenLtNode' : IR.NNNode.AtenLtNode.AtenLtNode(),
    'AtenMatmulNode' : IR.NNNode.AtenMatmulNode.AtenMatmulNode(),
    'AtenMaxNode' : IR.NNNode.AtenMaxNode.AtenMaxNode(),
    'AtenNeNode' : IR.NNNode.AtenNeNode.AtenNeNode(),
    'AtenNegNode' : IR.NNNode.AtenNegNode.AtenNegNode(),
    'AtenNotNode' : IR.NNNode.AtenNotNode.AtenNotNode(),
    'AtenReluNode' : IR.NNNode.AtenReluNode.AtenReluNode(),
    'AtenSubNode' : IR.NNNode.AtenSubNode.AtenSubNode(),
    'AtenTensorNode' : IR.NNNode.AtenTensorNode.AtenTensorNode(),
    'AtenZerosLikeNode' : IR.NNNode.AtenZerosLikeNode.AtenZerosLikeNode(),
    'AtenZerosNode' : IR.NNNode.AtenZerosNode.AtenZerosNode()
}


def create_prim_op_map():
    '''
    create a dict that contains all torch::prim Ops of RNNT
    '''
    op_dict = {
        IR.CONTROLNode.AnyType.AnyType().PrimBlockNode : 'PrimBlockNode',
        IR.CONTROLNode.AnyType.AnyType().PrimCallMethodNode : 'PrimCallMethodNode',
        IR.CONTROLNode.AnyType.AnyType().PrimConstantNode : 'PrimConstantNode',
        IR.CONTROLNode.AnyType.AnyType().PrimDataNode : 'PrimDataNode',
        IR.CONTROLNode.AnyType.AnyType().PrimDeviceNode : 'PrimDeviceNode',
        IR.CONTROLNode.AnyType.AnyType().PrimDtypeNode : 'PrimDtypeNode',
        IR.CONTROLNode.AnyType.AnyType().PrimEndIfNode : 'PrimEndIfNode',
        IR.CONTROLNode.AnyType.AnyType().PrimEndLoopNode : 'PrimEndLoopNode',
        IR.CONTROLNode.AnyType.AnyType().PrimIfNode : 'PrimIfNode',
        IR.CONTROLNode.AnyType.AnyType().PrimListConstructNode : 'PrimListConstructNode',
        IR.CONTROLNode.AnyType.AnyType().PrimListUnpackNode : 'PrimListUnpackNode',
        IR.CONTROLNode.AnyType.AnyType().PrimLoopIndexNode : 'PrimLoopIndexNode',
        IR.CONTROLNode.AnyType.AnyType().PrimLoopNode : 'PrimLoopNode',
        IR.CONTROLNode.AnyType.AnyType().PrimRaiseExceptionNode : 'PrimRaiseExceptionNode',
        IR.CONTROLNode.AnyType.AnyType().PrimTupleConstructNode : 'PrimTupleConstructNode',
        IR.CONTROLNode.AnyType.AnyType().PrimTupleIndexNode : 'PrimTupleIndexNode',
        IR.CONTROLNode.AnyType.AnyType().PrimTupleUnpackNode : 'PrimTupleUnpackNode',
        IR.CONTROLNode.AnyType.AnyType().PrimUncheckedCastNode : 'PrimUncheckedCastNode',
        IR.CONTROLNode.AnyType.AnyType().PrimUninitializedNode : 'PrimUninitializedNode'
    }
    return op_dict

prim_ops_dict = create_prim_op_map()

'''
these torch::prim Ops have no attribute
'''
prim_ops_no_attr_dict = {
    'PrimBlockNode' : IR.CONTROLNode.PrimBlockNode.PrimBlockNode(),
    'PrimDataNode' : IR.CONTROLNode.PrimDataNode.PrimDataNode(),
    'PrimDeviceNode' : IR.CONTROLNode.PrimDeviceNode.PrimDeviceNode(),
    'PrimDtypeNode' : IR.CONTROLNode.PrimDtypeNode.PrimDtypeNode(),
    'PrimEndIfNode' : IR.CONTROLNode.PrimEndIfNode.PrimEndIfNode(),
    'PrimEndLoopNode' : IR.CONTROLNode.PrimEndLoopNode.PrimEndLoopNode(),
    'PrimListConstructNode' : IR.CONTROLNode.PrimListConstructNode.PrimListConstructNode(),
    'PrimListUnpackNode' : IR.CONTROLNode.PrimListUnpackNode.PrimListUnpackNode(),
    'PrimRaiseExceptionNode' : IR.CONTROLNode.PrimRaiseExceptionNode.PrimRaiseExceptionNode(),
    'PrimTupleConstructNode' : IR.CONTROLNode.PrimTupleConstructNode.PrimTupleConstructNode(),
    'PrimTupleUnpackNode' : IR.CONTROLNode.PrimTupleUnpackNode.PrimTupleUnpackNode(),
    'PrimUncheckedCastNode' : IR.CONTROLNode.PrimUncheckedCastNode.PrimUncheckedCastNode(),
    'PrimUninitializedNode' : IR.CONTROLNode.PrimUninitializedNode.PrimUninitializedNode()
}

if __name__ =='__main__':
    print(prim_ops_dict)
    prim_list_construct = prim_ops_dict[IR.CONTROLNode.AnyType.AnyType().PrimListConstructNode]
    print(prim_list_construct)
