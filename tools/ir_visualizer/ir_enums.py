from enum import Enum


class DefaultEnum(Enum):
    @classmethod
    def get_name(cls, val):
        try:
            return cls(val).name
        except ValueError:
            return 'UNKNOWN'


# Enum Define
# 'IR' Enums
class NodeType(DefaultEnum):
    NNNode = 0
    OpNode = 1


class EdgeType(DefaultEnum):
    DataEdge = 0
    ControlEdge = 1


_TORCH_ATEN_OP_START = 100

class NNNodeType(DefaultEnum):
    NONE = 0
    InputNode = 1
    ConvNode = 2
    PoolNode = 3
    ActivationNode = 4
    FullyConnectedNode = 5
    ConcatNode = 6
    SoftmaxNode = 7
    EltwiseNode = 8
    BatchNormNode = 9
    ScaleNode = 10
    PriorBoxNode = 11
    PermuteNode = 12
    ReshapeNode = 13
    Pad = 14

    # Torch Aten Ops
    # for extension, Torch Aten Ops starts from "_TORCH_ATEN_OP_START"
    AtenZerosLikeNode = _TORCH_ATEN_OP_START + 0
    AtenZerosNode = _TORCH_ATEN_OP_START + 1
    AtenUnsqueezeNode = _TORCH_ATEN_OP_START + 2
    AtenToNode = _TORCH_ATEN_OP_START + 3
    AtenTensorNode = _TORCH_ATEN_OP_START + 4 
    AtenSliceNode = _TORCH_ATEN_OP_START + 5
    AtenSizeNode = _TORCH_ATEN_OP_START + 6
    AtenSelectNode = _TORCH_ATEN_OP_START + 7
    AtenNegNode = _TORCH_ATEN_OP_START + 8
    AtenNeNode = _TORCH_ATEN_OP_START + 9
    AtenLtNode = _TORCH_ATEN_OP_START + 10 
    AtenLSTMNode = _TORCH_ATEN_OP_START + 11
    AtenListNode = _TORCH_ATEN_OP_START + 12
    AtenLenNode = _TORCH_ATEN_OP_START + 13 
    AtenItemNode = _TORCH_ATEN_OP_START + 14
    AtenGtNode = _TORCH_ATEN_OP_START + 15
    AtenFormatNode = _TORCH_ATEN_OP_START + 16
    AtenExpandNode = _TORCH_ATEN_OP_START + 17
    AtenEqNode = _TORCH_ATEN_OP_START + 18
    AtenEmbeddingNode = _TORCH_ATEN_OP_START + 19
    AtenDropoutNode = _TORCH_ATEN_OP_START + 20
    AtenDivNode = _TORCH_ATEN_OP_START + 21 
    AtenDimNode = _TORCH_ATEN_OP_START + 22
    AtenCopyNode = _TORCH_ATEN_OP_START + 23
    AtenCeilNode = _TORCH_ATEN_OP_START + 24
    AtenCatNode = _TORCH_ATEN_OP_START + 25
    AtenAppendNode = _TORCH_ATEN_OP_START + 26
    AtenAdmmNode = _TORCH_ATEN_OP_START + 27
    AtenAddNode = _TORCH_ATEN_OP_START + 28
    AtenIsNode = _TORCH_ATEN_OP_START + 29
    AtenGetItemNode = _TORCH_ATEN_OP_START + 30
    AtenDeriveIndexNode = _TORCH_ATEN_OP_START + 31
    AtenIntNode = _TORCH_ATEN_OP_START + 32
    AtenAndNode = _TORCH_ATEN_OP_START + 33
    AtenPackPaddedSequenceNode = _TORCH_ATEN_OP_START + 34
    AtenPadPackedSequenceNode = _TORCH_ATEN_OP_START + 35
    AtenSetItemNode = _TORCH_ATEN_OP_START + 35
    AtenAnyNode = _TORCH_ATEN_OP_START + 36
    AtenArangeNode = _TORCH_ATEN_OP_START + 37
    AtenAsTensorNode = _TORCH_ATEN_OP_START + 38
    AtenBitwiseNotNode = _TORCH_ATEN_OP_START + 39
    AtenBmmNode = _TORCH_ATEN_OP_START + 39
    AtenBoolNode = _TORCH_ATEN_OP_START + 40
    AtenChunkNode = _TORCH_ATEN_OP_START + 41
    AtenClampNode = _TORCH_ATEN_OP_START + 42
    AtenClearNode = _TORCH_ATEN_OP_START + 43
    AtenContiguousNode = _TORCH_ATEN_OP_START + 44
    AtenCpuNode = _TORCH_ATEN_OP_START + 45
    AtenCudaNode = _TORCH_ATEN_OP_START + 46
    AtenEqualNode = _TORCH_ATEN_OP_START + 47
    AtenFloorDivideNode = _TORCH_ATEN_OP_START + 48
    AtenGeNode = _TORCH_ATEN_OP_START + 49
    AtenIndexNode = _TORCH_ATEN_OP_START + 50
    AtenFillNode = _TORCH_ATEN_OP_START + 51
    AtenLogNode = _TORCH_ATEN_OP_START + 52
    AtenTanhNode = _TORCH_ATEN_OP_START + 53
    AtenViewNode =  _TORCH_ATEN_OP_START + 54
    AtenMaskedSelectNode =  _TORCH_ATEN_OP_START + 55
    AtenPowNode =  _TORCH_ATEN_OP_START + 56


class ControlNodeType(DefaultEnum):
    NONE = 0
    PrimBlockNode = 1
    PrimConstantNode = 2
    PrimEndIfNode = 3
    PrimEndLoopNode = 4
    PrimGetAttrNode = 5
    PrimIf = 6
    PrimListUnpackNode = 7
    PrimLoopIndexNode = 8
    PrimLoopNode = 9
    PrimRaiseExceptionNode = 10
    PrimTupleConstructNode = 11
    PrimTupleIndexNode = 12
    PrimTupleUnpackNode = 13
    PrimUninitializedNode = 14
    PrimDataNode = 15
    PrimDeviceNode = 16
    PrimDtypeNode = 17
    primUncheckedCastNode = 18
    PrimSetAttrNode = 19

# 'NNNode' Enums
class InputType(DefaultEnum):
    Data = 0
    Image = 1
    Speech = 2
    Text = 3


class PoolType(DefaultEnum):
    Average = 0
    Max = 1
    Stochastic = 2


class ActivationType(DefaultEnum):
    None_ = 0
    Relu = 1
    Relu6 = 2
    Sigmoid = 3
    Tanh = 4
    LeakyRelu = 5
    PRelu = 6,
    PiecewiseLinear = 7,
    Relu1 = 8,
    Clip = 9


class PadCalculation(DefaultEnum):
    Ignore = 0
    Include = 1


class EltwiseType(DefaultEnum):
    Prod = 0
    Sum = 1
    Max = 2


# 'Type' Enums
class DataType(DefaultEnum):
    FP_32 = 0
    FIXED_16 = 1
    FIXED_8 = 2
    FIXED_8U = 3
    FIXED_64 = 4
    FIXED_16U = 5
    FIXED_32 = 6
    FIXED_32U = 7
    FP_16 = 8
    FIXED_4 = 9
    FIXED_4U = 10


class ShapeType(DefaultEnum):
    UNDEFINED = 0
    CELL = 1
    NCHW = 2
    NHWC = 3


class QuantType(DefaultEnum):
    SAIT = 0
    Google = 1


class NodeExecutionType(DefaultEnum):
    NODE_DATA_LOAD_START = 0
    NODE_DATA_LOAD_SYNC = 1
    EXEC_START = 2
    EXEC_SYNC = 3


class EdgeExecutionType(DefaultEnum):
    LOAD_START = 0
    LOAD_SYNC = 1
    STORE_START = 2
    STORE_SYNC = 3


# memory_instr
class DirectionType(DefaultEnum):
    DRAM2SRAM = 0
    SRAM2DRAM = 1
    DRAM2FIFO = 2


class MemoryType(DefaultEnum):
    DRAM = 0
    SRAM = 1
    FIFO = 2


class MemoryDataType(DefaultEnum):
    PSUM = 0
    IFM = 1
    KERNEL = 2
    OFM = 3
    INSTR = 4
    MISC = 5
    INTERMEDIATE_FM = 6
    SHARED_IM = 7
    CONSTANT = 8
    LUT = 9


class NodeOperationType(DefaultEnum):
    NORMAL = 0
    DEDICATED = 1


class TilingSchemeType(DefaultEnum):
    WEIGHT = 0
    IFM = 1
    IFM_WEIGHT = 2


class TilingDirectionType(DefaultEnum):
    OUTCHWISE = 0
    INCHWISE = 1
    HEIGHTWISE = 2
    WIDTHWISE = 3


class SyncType(DefaultEnum):
    NONE = 0
    LOCAL = 1
    REMOTE = 2


class SigType(DefaultEnum):
    NONE = 0
    Sig_wait = 1
    Sig_bar = 2
    Sig_send = 3


class PartitionMode(DefaultEnum):
    NONE = 0
    Branch = 1
    IFM_H = 2
    OFM_C = 3


class GlobalConcatType(DefaultEnum):
    Intra = 0
    Inter = 1


class GlobalConcatAxis(DefaultEnum):
    C_axis = 0
    H_axis = 1


class CompressionType(DefaultEnum):
    NONE = 0
    FLC = 1


class DataFormatConversion(DefaultEnum):
    RASTER2CELL = 0
    CELL2RASTER = 1


class PriorboxType(DefaultEnum):
    NONE = 0
    TF = 1


class PadType(DefaultEnum):
    ZERO = 0
    MIRROR = 1
    REPLICATE = 2
    CONSTANT = 3
    SYMMETRIC = 4
