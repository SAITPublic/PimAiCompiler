/*
 * Copyright (C) 2019 Samsung Electronics Co. LTD
 *
 * This software is proprietary of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

/**
 * @file.    ir_types.hpp
 * @brief.
 * @details. This header defines node/edge types for nn_ir::NNIR
 * @version. 0.1.
 */

#pragma once

#include "ir/include/common/log.hpp"
#include <ir/include/generated/ir_generated.h>

#include "common/include/arithmetics.hpp"
#include "common/include/attributes.h"
#include "common/include/cast.hpp"
#include "common/include/pretty_print.hpp"

#include "half.hpp"

#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

typedef half_float::half float16;
typedef int8_t           int4_t;
typedef uint8_t          uint4_t;

namespace nn_compiler {
namespace nn_ir {

#define GRAPH_ID_T      int32_t
#define NODE_ID_T       int32_t
#define EDGE_ID_T       int32_t
#define BLOB_ID_T       int32_t
#define INSTR_ID_T      int32_t
#define STEP_ID_T       int32_t
#define BIT_WIDTH_T     uint8_t
#define MEMORY_OFFSET_T int32_t
#define MEMORY_SIZE_T   int32_t
#define MEMORY_ID_T     uint32_t
#define FRAC_LENGTH_T   int8_t

#define INVALID_ID     -1
#define INVALID_OFFSET -1
#define INVALID_SIZE   -1

enum class DataType {
    FloatDataType = 0, // start describing float datatype
    FLOAT64,
    FLOAT32,
    FLOAT16,
    LastFloatDataType, // end describing nn nodes

    IntDataType, // start describing int datatype
    INT64,
    INT32,
    UINT32,
    INT16,
    UINT16,
    INT8,
    UINT8,
    INT4,
    UINT4,
    LastIntDataType, // start describing int datatype

    BOOL,
    DEVICE,
    LIST,
    STRING,
    TENSOR,
    NONE,
};

inline std::ostream& operator<<(std::ostream& s, DataType type) {
    switch (type) {
#define ENUM_STR(x)   \
    case DataType::x: \
        s << #x;      \
        break;
        ENUM_STR(FLOAT64)
        ENUM_STR(FLOAT32)
        ENUM_STR(FLOAT16)
        ENUM_STR(INT64)
        ENUM_STR(INT32)
        ENUM_STR(UINT32)
        ENUM_STR(INT16)
        ENUM_STR(UINT16)
        ENUM_STR(INT8)
        ENUM_STR(UINT8)
        ENUM_STR(INT4)
        ENUM_STR(UINT4)

        ENUM_STR(BOOL)
        ENUM_STR(DEVICE)
        ENUM_STR(LIST)
        ENUM_STR(STRING)
        ENUM_STR(TENSOR)
        ENUM_STR(NONE)
#undef ENUM_STR
        default:
            Log::IR::E() << "Invalid DataType " << static_cast<int>(type);
            break;
    }
    return s;
}

enum class QuantType { SYMMETRIC = 0, ASYMMETRIC = 1 };

enum class QuantLevelType { LAYERWISE = 0, CHANNELWISE = 1 };

enum class ShapeType { NCHW = 0, NHWC = 1 };

enum class Axis { N = 0, C = 1, H = 2, W = 3 };

/// @example for (const auto axis : nn_ir::AllAxes4D) { do_stuff[axis]; }
inline constexpr Axis AllAxes4D[] = {Axis::N, Axis::C, Axis::H, Axis::W};
inline constexpr Axis AllAxes3D[] = {Axis::C, Axis::H, Axis::W};
inline constexpr Axis AllAxes2D[] = {Axis::H, Axis::W};

inline std::ostream& operator<<(std::ostream& s, Axis axis) {
    switch (axis) {
        case Axis::N:
            s << 'N';
            return s;
        case Axis::C:
            s << 'C';
            return s;
        case Axis::H:
            s << 'H';
            return s;
        case Axis::W:
            s << 'W';
            return s;
        default:
            Log::IR::E() << "Unexpected axis was found";
    }
}

class NNIR;

class Node;
class Edge;
class Blob;
class Instruction;
class ExecutionStep;

class CONTROLNode;
class NNNode;
class HWNode;
class OPNode;
class QNode;
class VNode;
class GlobalNode;

class DataEdge;
class ControlEdge;

class ActivationNode;
class BatchNormNode;
class ConvolutionNode;
class DeconvolutionNode;
class EltwiseNode;
class InputNode;
class PoolNode;
class ConcatNode;
class SoftmaxNode;
class ScaleNode;
class FullyConnectedNode;
class ReshapeNode;
class DataFormatNode;
class PermuteNode;
class SliceNode;
class PriorBoxNode;
class TileNode;
class SpaceToDepthNode;
class DepthToSpaceNode;
class MatMulNode;
class DummyNode;
class CopyNode;

class AtenAddNode;
class AtenAddmmNode;
class AtenAndNode;
class AtenAnyNode;
class AtenAppendNode;
class AtenArange1Node;
class AtenArange2Node;
class AtenArange3Node;
class AtenAsTensorNode;
class AtenBitwiseNotNode;
class AtenBatchNorm2dNode;
class AtenBmmNode;
class AtenBoolNode;
class AtenCatNode;
class AtenCeilNode;
class AtenChunkNode;
class AtenClampNode;
class AtenClearNode;
class AtenContiguousNode;
class AtenConv2dNode;
class AtenCopyNode;
class AtenCpuNode;
class AtenCudaNode;
class AtenDeriveIndexNode;
class AtenDimNode;
class AtenDivNode;
class AtenDropoutNode;
class AtenEmbeddingNode;
class AtenEqNode;
class AtenEqualNode;
class AtenExpandNode;
class AtenFillNode;
class AtenFloorDivideNode;
class AtenFormatNode;
class AtenGatherNode;
class AtenGeNode;
class AtenGetItemNode;
class AtenGtNode;
class AtenIndexNode;
class AtenIndexPutNode;
class AtenIndexSelectNode;
class AtenIntNode;
class AtenIsNode;
class AtenItemNode;
class AtenLeakyReluNode;
class AtenLenNode;
class AtenLinearNode;
class AtenListNode;
class AtenLogNode;
class AtenLogSoftmaxNode;
class AtenLSTM1Node;
class AtenLSTM2Node;
class AtenLtNode;
class AtenMaskedFillNode;
class AtenMaskedSelectNode;
class AtenMatmulNode;
class AtenMaxNode;
class AtenMaxPool2dNode;
class AtenMinNode;
class AtenMulNode;
class AtenNeNode;
class AtenNegNode;
class AtenNormNode;
class AtenNotNode;
class AtenOnesNode;
class AtenPackPaddedSequenceNode;
class AtenPadPackedSequenceNode;
class AtenPowNode;
class AtenReluNode;
class AtenReshapeNode;
class AtenSelectNode;
class AtenSetItemNode;
class AtenSizeNode;
class AtenSliceNode;
class AtenSoftmaxNode;
class AtenSqueezeNode;
class AtenSubNode;
class AtenSumNode;
class AtenTanhNode;
class AtenTensorNode;
class AtenTo1Node;
class AtenTo2Node;
class AtenTopkNode;
class AtenTransposeNode;
class AtenUnsqueezeNode;
class AtenViewNode;
class AtenWarnNode;
class AtenZerosLikeNode;
class AtenZerosNode;

class ShiftNode;

class QuantNode;
class DequantNode;

class GlobalSplitNode;
class GlobalConcatNode;
class GlobalSyncNode;

class VConcatNode;
class VSplitNode;

class VLoadNode;
class VStoreNode;

class ComputeInstruction;
class MemoryInstruction;
class MiscInstruction;

class DMAStartInstruction;
class DMASyncInstruction;

class ExecuteStartInstruction;
class ExecuteSyncInstruction;

class SignalSendInstruction;
class SignalWaitInstruction;

class NodeExecutionStep;
class EdgeExecutionStep;

class MAAEltwiseNode;

class PrimBlockNode;
class PrimConstantNode;
class PrimDataNode;
class PrimDeviceNode;
class PrimDtypeNode;
class PrimEndIfNode;
class PrimEndLoopNode;
class PrimGetAttrNode;
class PrimIfNode;
class PrimInputNode;
class PrimListConstructNode;
class PrimListUnpackNode;
class PrimLoopIndexNode;
class PrimLoopNode;
class PrimOutputNode;
class PrimRaiseExceptionNode;
class PrimSetAttrNode;
class PrimTupleConstructNode;
class PrimTupleIndexNode;
class PrimTupleUnpackNode;
class PrimTypeNode;
class PrimUncheckedCastNode;
class PrimUninitializedNode;
class PrimVariableNode;

enum class NodeType {
    CONTROLNode, // start describing control nodes
    PRIMBLOCK,
    PRIMCONSTANT,
    PRIMDATA,
    PRIMDEVICE,
    PRIMDTYPE,
    PRIMENDIF,
    PRIMENDLOOP,
    PRIMGETATTR,
    PRIMIF,
    PRIMINPUT,
    PRIMLISTCONSTRUCT,
    PRIMLISTUNPACK,
    PRIMLOOPINDEX,
    PRIMLOOP,
    PRIMOUTPUT,
    PRIMRAISEEXCEPTION,
    PRIMSETATTR,
    PRIMTUPLECONSTRUCT,
    PRIMTUPLEINDEX,
    PRIMTUPLEUNPACK,
    PRIMTYPE,
    PRIMUNCHECKEDCAST,
    PRIMUNINITIALIZED,
    PRIMVARIABLE,
    LastCONTROLNode, // end describing control nodes

    NNNode, // start describing nn nodes
    INPUT,
    CONVOLUTION,
    ACTIVATION,
    POOL,
    FULLYCONNECTED,
    CONCAT,
    SOFTMAX,
    ELTWISE,
    BATCHNORM,
    SCALE,
    DECONVOLUTION,
    RESHAPE,
    DATAFORMAT,
    PERMUTE,
    PRIORBOX,
    SLICE,
    TILE,
    SPACETODEPTH,
    DEPTHTOSPACE,
    MATMUL,
    DUMMY,
    COPY,

    ATENADD,
    ATENADDMM,
    ATENAND,
    ATENANY,
    ATENAPPEND,
    ATENARANGE1,
    ATENARANGE2,
    ATENARANGE3,
    ATENASTENSOR,
    ATENBATCHNORM2D,
    ATENBITWISENOT,
    ATENBMM,
    ATENBOOL,
    ATENCAT,
    ATENCEIL,
    ATENCHUNK,
    ATENCLAMP,
    ATENCLEAR,
    ATENCONTIGUOUS,
    ATENCONV2D,
    ATENCOPY,
    ATENCPU,
    ATENCUDA,
    ATENDERIVEINDEX,
    ATENDIM,
    ATENDIV,
    ATENDROPOUT,
    ATENEMBEDDING,
    ATENEQ,
    ATENEQUAL,
    ATENEXPAND,
    ATENFILL,
    ATENFLOORDIVIDE,
    ATENFORMAT,
    ATENGATHER,
    ATENGE,
    ATENGETITEM,
    ATENGT,
    ATENINDEX,
    ATENINDEXPUT,
    ATENINDEXSELECT,
    ATENINT,
    ATENIS,
    ATENITEM,
    ATENLEAKYRELU,
    ATENLEN,
    ATENLINEAR,
    ATENLIST,
    ATENLOG,
    ATENLOGSOFTMAX,
    ATENLSTM1,
    ATENLSTM2,
    ATENLT,
    ATENMASKEDFILL,
    ATENMASKEDSELECT,
    ATENMATMUL,
    ATENMAX,
    ATENMAXPOOL2D,
    ATENMIN,
    ATENMUL,
    ATENNE,
    ATENNEG,
    ATENNORM,
    ATENNOT,
    ATENONES,
    ATENPACKPADDEDSEQUENCE,
    ATENPADPACKEDSEQUENCE,
    ATENPOW,
    ATENRELU,
    ATENRESHAPE,
    ATENSELECT,
    ATENSETITEM,
    ATENSIZE,
    ATENSLICE,
    ATENSOFTMAX,
    ATENSQUEEZE,
    ATENSUB,
    ATENSUM,
    ATENTANH,
    ATENTENSOR,
    ATENTO1,
    ATENTO2,
    ATENTOPK,
    ATENTRANSPOSE,
    ATENUNSQUEEZE,
    ATENVIEW,
    ATENWARN,
    ATENZEROS,
    ATENZEROSLIKE,
    LastNNNode, // end describing nn nodes

    OPNode, // start describing op nodes
    SHIFT,
    LastOPNode, // end describing op nodes

    VNode, // start describing virtual nodes
    VCONCAT,
    VSPLIT,
    LastVNode, // end describing virtual nodes

    GlobalNode, // start describing global nodes
    GSPLIT,
    GCONCAT,
    GSYNC,
    LastGlobalNode, // end describing global nodes

    QNode, // start describing quant nodes
    QUANT,
    DEQUANT,
    LastQNode, // end describing quant nodes

    HWNode, // start describing hw nodes
    MAAELTWISE,
    LastHWNode // end describing hw nodes
};

inline std::ostream& operator<<(std::ostream& s, nn_ir::NodeType type) {
    switch (type) {
#define ENUM_STR(x)          \
    case nn_ir::NodeType::x: \
        s << #x;             \
        break;
        ENUM_STR(INPUT)
        ENUM_STR(CONVOLUTION)
        ENUM_STR(ACTIVATION)
        ENUM_STR(POOL)
        ENUM_STR(FULLYCONNECTED)
        ENUM_STR(CONCAT)
        ENUM_STR(SOFTMAX)
        ENUM_STR(ELTWISE)
        ENUM_STR(BATCHNORM)
        ENUM_STR(SCALE)
        ENUM_STR(DECONVOLUTION)
        ENUM_STR(RESHAPE)
        ENUM_STR(DATAFORMAT)
        ENUM_STR(PERMUTE)
        ENUM_STR(PRIORBOX)
        ENUM_STR(SLICE)
        ENUM_STR(TILE)
        ENUM_STR(SPACETODEPTH)
        ENUM_STR(DEPTHTOSPACE)
        ENUM_STR(DUMMY)
        ENUM_STR(COPY)
        ENUM_STR(SHIFT)
        ENUM_STR(VCONCAT)
        ENUM_STR(VSPLIT)
        ENUM_STR(GSPLIT)
        ENUM_STR(GCONCAT)
        ENUM_STR(GSYNC)
        ENUM_STR(QUANT)
        ENUM_STR(DEQUANT)
        ENUM_STR(MATMUL)
        ENUM_STR(MAAELTWISE)

        ENUM_STR(ATENADD)
        ENUM_STR(ATENADDMM)
        ENUM_STR(ATENAND)
        ENUM_STR(ATENANY)
        ENUM_STR(ATENAPPEND)
        ENUM_STR(ATENARANGE1)
        ENUM_STR(ATENARANGE2)
        ENUM_STR(ATENARANGE3)
        ENUM_STR(ATENASTENSOR)
        ENUM_STR(ATENBATCHNORM2D)
        ENUM_STR(ATENBITWISENOT)
        ENUM_STR(ATENBMM)
        ENUM_STR(ATENBOOL)
        ENUM_STR(ATENCAT)
        ENUM_STR(ATENCEIL)
        ENUM_STR(ATENCHUNK)
        ENUM_STR(ATENCLAMP)
        ENUM_STR(ATENCLEAR)
        ENUM_STR(ATENCONTIGUOUS)
        ENUM_STR(ATENCONV2D)
        ENUM_STR(ATENCOPY)
        ENUM_STR(ATENCPU)
        ENUM_STR(ATENCUDA)
        ENUM_STR(ATENDERIVEINDEX)
        ENUM_STR(ATENDIM)
        ENUM_STR(ATENDIV)
        ENUM_STR(ATENDROPOUT)
        ENUM_STR(ATENEMBEDDING)
        ENUM_STR(ATENEQ)
        ENUM_STR(ATENEQUAL)
        ENUM_STR(ATENEXPAND)
        ENUM_STR(ATENFILL)
        ENUM_STR(ATENFLOORDIVIDE)
        ENUM_STR(ATENFORMAT)
        ENUM_STR(ATENGATHER)
        ENUM_STR(ATENGE)
        ENUM_STR(ATENGETITEM)
        ENUM_STR(ATENGT)
        ENUM_STR(ATENINDEX)
        ENUM_STR(ATENINDEXPUT)
        ENUM_STR(ATENINDEXSELECT)
        ENUM_STR(ATENINT)
        ENUM_STR(ATENIS)
        ENUM_STR(ATENITEM)
        ENUM_STR(ATENLEAKYRELU)
        ENUM_STR(ATENLEN)
        ENUM_STR(ATENLINEAR)
        ENUM_STR(ATENLIST)
        ENUM_STR(ATENLOG)
        ENUM_STR(ATENLOGSOFTMAX)
        ENUM_STR(ATENLSTM1)
        ENUM_STR(ATENLSTM2)
        ENUM_STR(ATENLT)
        ENUM_STR(ATENMASKEDFILL)
        ENUM_STR(ATENMASKEDSELECT)
        ENUM_STR(ATENMATMUL)
        ENUM_STR(ATENMAX)
        ENUM_STR(ATENMAXPOOL2D)
        ENUM_STR(ATENMIN)
        ENUM_STR(ATENMUL)
        ENUM_STR(ATENNE)
        ENUM_STR(ATENNEG)
        ENUM_STR(ATENNORM)
        ENUM_STR(ATENNOT)
        ENUM_STR(ATENONES)
        ENUM_STR(ATENPACKPADDEDSEQUENCE)
        ENUM_STR(ATENPADPACKEDSEQUENCE)
        ENUM_STR(ATENPOW)
        ENUM_STR(ATENRELU)
        ENUM_STR(ATENRESHAPE)
        ENUM_STR(ATENSELECT)
        ENUM_STR(ATENSETITEM)
        ENUM_STR(ATENSIZE)
        ENUM_STR(ATENSLICE)
        ENUM_STR(ATENSOFTMAX)
        ENUM_STR(ATENSQUEEZE)
        ENUM_STR(ATENSUB)
        ENUM_STR(ATENSUM)
        ENUM_STR(ATENTANH)
        ENUM_STR(ATENTENSOR)
        ENUM_STR(ATENTO1)
        ENUM_STR(ATENTO2)
        ENUM_STR(ATENTOPK)
        ENUM_STR(ATENTRANSPOSE)
        ENUM_STR(ATENUNSQUEEZE)
        ENUM_STR(ATENVIEW)
        ENUM_STR(ATENWARN)
        ENUM_STR(ATENZEROS)
        ENUM_STR(ATENZEROSLIKE)

        ENUM_STR(PRIMBLOCK)
        ENUM_STR(PRIMCONSTANT)
        ENUM_STR(PRIMDATA)
        ENUM_STR(PRIMDEVICE)
        ENUM_STR(PRIMDTYPE)
        ENUM_STR(PRIMENDIF)
        ENUM_STR(PRIMENDLOOP)
        ENUM_STR(PRIMGETATTR)
        ENUM_STR(PRIMIF)
        ENUM_STR(PRIMINPUT)
        ENUM_STR(PRIMLISTCONSTRUCT)
        ENUM_STR(PRIMLISTUNPACK)
        ENUM_STR(PRIMLOOPINDEX)
        ENUM_STR(PRIMLOOP)
        ENUM_STR(PRIMOUTPUT)
        ENUM_STR(PRIMRAISEEXCEPTION)
        ENUM_STR(PRIMSETATTR)
        ENUM_STR(PRIMTUPLECONSTRUCT)
        ENUM_STR(PRIMTUPLEINDEX)
        ENUM_STR(PRIMTUPLEUNPACK)
        ENUM_STR(PRIMTYPE)
        ENUM_STR(PRIMUNCHECKEDCAST)
        ENUM_STR(PRIMUNINITIALIZED)
        ENUM_STR(PRIMVARIABLE)

#undef ENUM_STR
        default:
            Log::IR::E() << "Invalid opcode " << static_cast<int>(type);
            break;
    }
    return s;
}

enum class BlobType { FEATUREMAP = 0, WEIGHT = 1, BIAS = 2, LUT = 3 };

inline std::ostream& operator<<(std::ostream& s, BlobType type) {
    switch (type) {
#define ENUM_STR(x)   \
    case BlobType::x: \
        s << #x;      \
        break;
        ENUM_STR(FEATUREMAP)
        ENUM_STR(WEIGHT)
        ENUM_STR(BIAS)
        ENUM_STR(LUT)
#undef ENUM_STR
        default:
            Log::IR::E() << "Invalid BlobType " << static_cast<int>(type);
            break;
    }
    return s;
}

enum class TypeOfBlob { IFM = 0, OFM, WEIGHT, BIAS, COUNT };
constexpr std::size_t COUNT_TYPE_OF_BLOB         = static_cast<std::size_t>(nn_ir::TypeOfBlob::COUNT);
constexpr std::size_t COUNT_TYPE_OF_BLOB_NO_BIAS = static_cast<std::size_t>(nn_ir::TypeOfBlob::BIAS);

enum class EdgeType {
    DATA    = 0,
    CONTROL = 1,
};

enum class ActivationType {
    NONE            = 0,
    RELU            = 1,
    RELU6           = 2,
    SIGMOID         = 3,
    TANH            = 4,
    LEAKY_RELU      = 5,
    PRELU           = 6,
    PIECEWISELINEAR = 7,
    RELU1           = 8,
    CLIP            = 9,
    COUNT           = 10
};

enum class EltwiseType { PROD = 0, SUM = 1, MAX = 2 };

enum class PoolType { AVERAGE = 0, MAX = 1, STOCHASTIC = 2 };

enum class PadCalcType { IGNORE = 0, INCLUDE = 1 };

enum class PriorboxType { NONE = 0, TF = 1 };

enum class InputType { DATA = 0, IMAGE = 1, SPEECH = 2, TEXT = 3 };

enum class DataFormatConversion { TENSOR2CELL = 0, CELL2TENSOR = 1 };

inline std::ostream& operator<<(std::ostream& s, DataFormatConversion dir) {
    switch (dir) {
#define ENUM_STR(x)               \
    case DataFormatConversion::x: \
        s << #x;                  \
        break;
        ENUM_STR(TENSOR2CELL)
        ENUM_STR(CELL2TENSOR)
#undef ENUM_STR
        default:
            Log::IR::E() << "Invalid DataFormat direction " << static_cast<int>(dir);
            break;
    }
    return s;
}

enum class InstructionType {
    MEM_INSTR,
    DMA_START,
    DMA_SYNC,
    Last_MEM_INSTR,
    COMP_INSTR,
    COMPUTE_START,
    COMPUTE_SYNC,
    Last_COMP_INSTR,
    MISC_INSTR,
    SIG_SEND,
    SIG_WAIT,
    VSYNC,
    Last_MISC_INSTR
};

inline std::ostream& operator<<(std::ostream& s, InstructionType type) {
    switch (type) {
#define ENUM_STR(x)          \
    case InstructionType::x: \
        s << #x;             \
        break;
        ENUM_STR(MEM_INSTR)
        ENUM_STR(DMA_START)
        ENUM_STR(DMA_SYNC)
        ENUM_STR(COMPUTE_START)
        ENUM_STR(COMPUTE_SYNC)
        ENUM_STR(SIG_SEND)
        ENUM_STR(SIG_WAIT)
        ENUM_STR(VSYNC)
#undef ENUM_STR
        default:
            Log::IR::E() << "Invalid InstructionType " << static_cast<int>(type);
            break;
    }
    return s;
}

enum class ExecutionStepType { NODE = 0, EDGE = 1 };

inline std::ostream& operator<<(std::ostream& s, ExecutionStepType type) {
    switch (type) {
#define ENUM_STR(x)            \
    case ExecutionStepType::x: \
        s << #x;               \
        break;
        ENUM_STR(NODE)
        ENUM_STR(EDGE)
#undef ENUM_STR
        default:
            Log::IR::E() << "Invalid ExecutionStepType " << static_cast<int>(type);
            break;
    }
    return s;
}

enum class NodeExecutionStepType {
    NODE_DATA_LOAD_START = 0,
    NODE_DATA_LOAD_SYNC  = 1,
    EXEC_START           = 2,
    EXEC_SYNC            = 3,
    COUNT
};

static inline std::ostream& operator<<(std::ostream& s, NodeExecutionStepType step) {
    switch (step) {
        case NodeExecutionStepType::NODE_DATA_LOAD_START:
            s << "load weights";
            break;
        case NodeExecutionStepType::NODE_DATA_LOAD_SYNC:
            s << "sync weights";
            break;
        case NodeExecutionStepType::EXEC_START:
            s << "exec";
            break;
        case NodeExecutionStepType::EXEC_SYNC:
            s << "sync exec";
            break;
        default:
            Log::IR::E() << "Unknown NodeExecutionStepType " << static_cast<int>(step);
    }
    return s;
}

enum class EdgeExecutionStepType { LOAD_START = 0, LOAD_SYNC = 1, STORE_START = 2, STORE_SYNC = 3, COUNT };

static inline std::ostream& operator<<(std::ostream& s, EdgeExecutionStepType step) {
    switch (step) {
        case EdgeExecutionStepType::LOAD_START:
            s << "load ifm";
            break;
        case EdgeExecutionStepType::LOAD_SYNC:
            s << "sync ifm";
            break;
        case EdgeExecutionStepType::STORE_START:
            s << "store ofm";
            break;
        case EdgeExecutionStepType::STORE_SYNC:
            s << "sync ofm";
            break;
        default:
            Log::IR::E() << "Unknown EdgeExecutionStepType " << static_cast<int>(step);
    }
    return s;
}

enum class DMADirection {
    DRAM2SRAM = 0,
    SRAM2DRAM = 1,
    SRAM2SRAM = 2,
    DRAM2FIFO = 3,
};

inline std::ostream& operator<<(std::ostream& s, DMADirection direction) {
    switch (direction) {
#define ENUM_STR(x)       \
    case DMADirection::x: \
        s << #x;          \
        break;
        ENUM_STR(DRAM2SRAM)
        ENUM_STR(SRAM2DRAM)
        ENUM_STR(DRAM2FIFO)
#undef ENUM_STR
        default:
            Log::IR::E() << "Invalid DMADirection " << static_cast<int>(direction);
            break;
    }
    return s;
}

enum class GlobalSyncDirection {
    NPU2DSP = 0,
    DSP2NPU = 1,
};

enum class Core { Core0 = 0, Core1 = 1, Core2 = 2, Core3 = 3, COUNT };

/**
 * @brief   Here are two mixin classes Base2D and Base4D that are used by common structures
 *          that have dimensional fields and common methods. CRTP is used
 *          here is to prevent unwanted castings from one derived class to another one,
 *          i.e., by means of static_cast, that would lead to unexpected and incorrect behavior.
 * @warning Be aware when you use aggregate initialization. As soon as Coord4D, Shape4D,
 *          TilePosition and etc. have started using CRTP, now theirs aggregate initialization
 *          is set in another way with doubled braces {{...}}.
 * @warning All classes that derive Base4D are obliged to use a trait mask that allows to enable
 *          methods for some derived class. A static_assert checks if the trait mask of the derived class
 *          contains unnecessary bit - if it did not imply that a class is able to invoke
 *          Base4D method, then a compilation error will appear.
 */

template <class DimT>
struct Base2D {
    uint32_t h = 0;
    uint32_t w = 0;

#define CHECK_DERIVED_CLASS_VALIDNESS \
    static_assert(std::is_base_of<Base2D, DimT>::value, "This class must be used via CRTP");

    bool isValid() const {
        CHECK_DERIVED_CLASS_VALIDNESS
        return w && h;
    }

    friend bool operator==(const DimT& lhs, const DimT& rhs) {
        CHECK_DERIVED_CLASS_VALIDNESS
        return lhs.h == rhs.h && lhs.w == rhs.w;
    }
    friend bool operator!=(const DimT& lhs, const DimT& rhs) {
        CHECK_DERIVED_CLASS_VALIDNESS
        return !(lhs == rhs);
    }
    friend bool operator>=(const DimT& lhs, const DimT& rhs) {
        CHECK_DERIVED_CLASS_VALIDNESS
        return lhs.h >= rhs.h && lhs.w >= rhs.w;
    }
    friend bool operator<=(const DimT& lhs, const DimT& rhs) {
        CHECK_DERIVED_CLASS_VALIDNESS
        return lhs.h <= rhs.h && lhs.w <= rhs.w;
    }
    friend bool operator>(const DimT& lhs, const DimT& rhs) {
        CHECK_DERIVED_CLASS_VALIDNESS
        return !(lhs <= rhs);
    }
    friend bool operator<(const DimT& lhs, const DimT& rhs) {
        CHECK_DERIVED_CLASS_VALIDNESS
        return !(lhs >= rhs);
    }

    uint32_t getNumberOfElements() const {
        CHECK_DERIVED_CLASS_VALIDNESS
        return w * h;
    }

    const uint32_t& operator[](Axis axis) const {
        CHECK_DERIVED_CLASS_VALIDNESS
        switch (axis) {
            case Axis::H:
                return h;
            case Axis::W:
                return w;
            default:
                Log::IR::E() << "Unexpected axis was found";
        }
    }
    uint32_t& operator[](Axis axis) {
        CHECK_DERIVED_CLASS_VALIDNESS
        return const_cast<uint32_t&>(static_cast<const Base2D<DimT>&>(*this)[axis]);
    }
#undef CHECK_DERIVED_CLASS_VALIDNESS
};

struct Shape4D;

enum class GlobalConcatAxis { C_axis = 0, H_axis = 1 };

enum class GlobalConcatType { INTRA = 0, INTER = 1 };

enum class PartitionMode { NONE = 0, BRANCH = 1, IFM_H = 2, OFM_C = 3 };

enum class TilingScheme {
    NONE       = 0,
    WEIGHT     = 1,
    IFM        = 2,
    IFM_WEIGHT = 3,
};

template <class DimT, class ShapeT = Shape4D>
struct Base4D {
    uint32_t n = 0;
    uint32_t c = 0;
    uint32_t h = 0;
    uint32_t w = 0;

    using TraitType = uint32_t;

    enum : TraitType {
        FIRST_TRAIT                = 1 << 0,
        USE_NO_EXTRA_METHODS_TRAIT = FIRST_TRAIT,
        USE_ALIGN_METHODS_TRAIT    = 1 << 1,
        USE_NON_ZERO_FIELDS        = 1 << 2,
        LAST_TRAIT                 = USE_NON_ZERO_FIELDS | USE_ALIGN_METHODS_TRAIT | USE_NO_EXTRA_METHODS_TRAIT
    };

#define CHECK_DERIVED_CLASS_VALIDNESS                                                        \
    static_assert(std::is_base_of<Base4D, DimT>::value, "This class must be used via CRTP"); \
    static_assert(std::is_same_v<TraitType, std::decay_t<decltype(DimT::trait_mask)>>,       \
                  "Type of trait mask of this class is invalid");                            \
    static_assert(DimT::trait_mask >= FIRST_TRAIT && DimT::trait_mask <= LAST_TRAIT,         \
                  "Incorrect values in trait mask of this class");

    bool isValid() const {
        CHECK_DERIVED_CLASS_VALIDNESS
        static_assert(DimT::trait_mask | USE_NON_ZERO_FIELDS,
                      "This class cannot use this method. Look at trait mask of this class");
        return n && c && h && w;
    }

    friend bool operator==(const DimT& lhs, const DimT& rhs) {
        CHECK_DERIVED_CLASS_VALIDNESS
        return lhs.n == rhs.n && lhs.c == rhs.c && lhs.h == rhs.h && lhs.w == rhs.w;
    }

    friend bool operator!=(const DimT& lhs, const DimT& rhs) {
        CHECK_DERIVED_CLASS_VALIDNESS
        return !(lhs == rhs);
    }

    friend bool operator>=(const DimT& lhs, const DimT& rhs) {
        CHECK_DERIVED_CLASS_VALIDNESS
        return lhs.n >= rhs.n && lhs.c >= rhs.c && lhs.h >= rhs.h && lhs.w >= rhs.w;
    }

    friend bool operator<=(const DimT& lhs, const DimT& rhs) {
        CHECK_DERIVED_CLASS_VALIDNESS
        return lhs.n <= rhs.n && lhs.c <= rhs.c && lhs.h <= rhs.h && lhs.w <= rhs.w;
    }

    friend bool operator>(const DimT& lhs, const DimT& rhs) {
        CHECK_DERIVED_CLASS_VALIDNESS
        return !(lhs <= rhs);
    }

    friend bool operator<(const DimT& lhs, const DimT& rhs) {
        CHECK_DERIVED_CLASS_VALIDNESS
        return !(lhs >= rhs);
    }

    friend DimT operator+(DimT lhs, const DimT& rhs) {
        CHECK_DERIVED_CLASS_VALIDNESS
        lhs += rhs;
        return lhs;
    }

    Base4D<DimT>& operator+=(const Base4D<DimT>& rhs) {
        CHECK_DERIVED_CLASS_VALIDNESS
        w += rhs.w;
        h += rhs.h;
        c += rhs.c;
        n += rhs.n;
        return *this;
    }

    DimT getAlignedUpBy(const ShapeT& alignment_unit) const {
        CHECK_DERIVED_CLASS_VALIDNESS
        static_assert(DimT::trait_mask & USE_ALIGN_METHODS_TRAIT,
                      "This class cannot use this method. Look at trait mask of this class");
        return {{.n = alignUp(n, alignment_unit.n),
                 .c = alignUp(c, alignment_unit.c),
                 .h = alignUp(h, alignment_unit.h),
                 .w = alignUp(w, alignment_unit.w)}};
    }
    DimT getAlignedDownBy(const ShapeT& alignment_unit) const {
        CHECK_DERIVED_CLASS_VALIDNESS
        static_assert(DimT::trait_mask & USE_ALIGN_METHODS_TRAIT,
                      "This class cannot use this method. Look at trait mask of this class");
        return {{.n = alignDown(n, alignment_unit.n),
                 .c = alignDown(c, alignment_unit.c),
                 .h = alignDown(h, alignment_unit.h),
                 .w = alignDown(w, alignment_unit.w)}};
    }

    bool isTooLarge() const {
        CHECK_DERIVED_CLASS_VALIDNESS
        auto max = std::numeric_limits<std::common_type_t<decltype(n), decltype(c), decltype(h), decltype(w)>>::max();
        return ((n > max / c) || (h > max / w) || ((n * c) > max / (h * w)));
    }

    auto getNumberOfElements() const {
        CHECK_DERIVED_CLASS_VALIDNESS
        // for the situation that dsp takes all the task of this node.
        // npu only need to wait for the dsp.
        if (n * c * h * w == 0) {
            Log::IR::D() << "Shape with 0 situation found";
            return 0u;
        }
        Log::IR::E_IF(isTooLarge()) << "A tensor of too large shape was found";
        return n * c * h * w;
    }

    uint32_t getNumberOfElementsAligned(const Shape4D& alignment_unit) const {
        CHECK_DERIVED_CLASS_VALIDNESS
        return getAlignedUpBy(alignment_unit).getNumberOfElements();
    }

    const uint32_t& operator[](const Axis& axis) const {
        CHECK_DERIVED_CLASS_VALIDNESS
        switch (axis) {
            case Axis::N:
                return n;
            case Axis::C:
                return c;
            case Axis::H:
                return h;
            case Axis::W:
                return w;
            default:
                Log::IR::E() << "Unexpected axis was found";
        }
    }
    uint32_t& operator[](const Axis& axis) {
        CHECK_DERIVED_CLASS_VALIDNESS
        return const_cast<uint32_t&>(static_cast<const Base4D<DimT>&>(*this)[axis]);
    }
#undef CHECK_DERIVED_CLASS_VALIDNESS
};

/// @brief Describes padding projected onto single spatial axis (H or W)
typedef struct Pad1D_ {
    uint32_t front;
    uint32_t back;

    friend bool operator==(const Pad1D_& lhs, const Pad1D_& rhs) {
        return lhs.front == rhs.front && lhs.back == rhs.back;
    }
    friend bool operator!=(const Pad1D_& lhs, const Pad1D_& rhs) { return !(lhs == rhs); }
} Pad1D;

inline std::ostream& operator<<(std::ostream& s, const Pad1D& pad) {
    s << "{front:" << pad.front << " back:" << pad.back << '}';
    return s;
}

/// @brief Describes 2D spatial padding that adds zeros around HxW area of the feature-map
/// @warning Special use-cases are presented such as Deconvolution nodes where it's responsible for OFM trimming
typedef struct Pad4_ {
    uint32_t t; /// top
    uint32_t b; /// bottom
    uint32_t l; /// left
    uint32_t r; /// right

    bool isAllZeros() const { return t == 0 && b == 0 && l == 0 && r == 0; }

    friend bool operator==(const Pad4_& lhs, const Pad4_& rhs) {
        return lhs.t == rhs.t && lhs.b == rhs.b && lhs.l == rhs.l && lhs.r == rhs.r;
    }
    friend bool operator!=(const Pad4_& lhs, const Pad4_& rhs) { return !(lhs == rhs); }
    Pad1D       operator[](Axis axis) const {
        switch (axis) {
            case Axis::H:
                return {.front = t, .back = b};
            case Axis::W:
                return {.front = l, .back = r};
            default:
                Log::IR::E() << "Unexpected axis was found";
        }
    }
} Pad4;

inline std::ostream& operator<<(std::ostream& s, const Pad4& pad) {
    s << "{t:" << pad.t << " b:" << pad.b << " l:" << pad.l << " r:" << pad.r << '}';
    return s;
}

/**
 * @brief Describes numbers of tile-segments in the layout along 4 axes
 * @example {n:1, c:2, h:3, w:1}  would mean "tensor was split into 6 tiles: cut in 2 pieces depthwise and 3 heightwise"
 */
struct TileNumbers : Base4D<TileNumbers> {
    static constexpr TraitType trait_mask = USE_NO_EXTRA_METHODS_TRAIT;

    bool hasSpatialTiles() const { return h > 1 || w > 1; }
    bool hasChannelwiseTiles() const { return c > 1; }
    bool hasBatchwiseTiles() const { return n > 1; }

    bool hasSpatialTilesOnly() const { return hasSpatialTiles() && !hasBatchwiseTiles() && !hasChannelwiseTiles(); }
    bool hasChannelwiseTilesOnly() const { return hasChannelwiseTiles() && !hasBatchwiseTiles() && !hasSpatialTiles(); }
    bool hasBatchwiseTilesOnly() const { return hasBatchwiseTiles() && !hasSpatialTiles() && !hasChannelwiseTiles(); }

    auto getNumberOfTiles() const { return getNumberOfElements(); }
};

inline std::ostream& operator<<(std::ostream& s, const TileNumbers& numbers) {
    s << "{n:" << numbers.n << " c:" << numbers.c << " h:" << numbers.h << " w:" << numbers.w << '}';
    return s;
}

/**
 * @brief Describes 2D shape of a tensor: h and w only
 * @warning Length along each dimension should be non-zero
 */
struct Shape2D : Base2D<Shape2D> {
    Shape2D getDilated(const nn_ir::Shape2D& dilation) {
        return {{.h = applyDilation(h, dilation.h), .w = applyDilation(w, dilation.w)}};
    }
};

inline std::ostream& operator<<(std::ostream& s, const Shape2D& shape) {
    s << "[h:" << shape.h << " w:" << shape.w << ']';
    return s;
}

/**
 * @brief Describes 4D shape of a tensor: n, c, h, w only
 * @warning Length along each dimension should be non-zero
 */
struct Shape4D : Base4D<Shape4D> {
    static constexpr TraitType trait_mask = USE_ALIGN_METHODS_TRAIT | USE_NON_ZERO_FIELDS;

    TileNumbers getIndexOf(const Shape4D& base_unit) const {
        return {{.n = divUp(n, base_unit.n),
                 .c = divUp(c, base_unit.c),
                 .h = divUp(h, base_unit.h),
                 .w = divUp(w, base_unit.w)}};
    }

    friend Shape4D operator+(Shape4D lhs, const Pad4& rhs) {
        lhs += rhs;
        return lhs;
    }

    Shape4D& operator+=(const Pad4& rhs) {
        w += rhs.l + rhs.r;
        h += rhs.t + rhs.b;
        return *this;
    }

    Shape2D getShape2D() const { return {{.h = h, .w = w}}; }
};

inline std::ostream& operator<<(std::ostream& s, const Shape4D& shape) {
    s << "[n:" << shape.n << " c:" << shape.c << " h:" << shape.h << " w:" << shape.w << ']';
    return s;
}

/**
 * @brief Describes 4D coordinates of a point (value, pixel) within some 4D tensor
 * @warning Indicing is zero-based, so each coordinate is valid in range [0, (tensor_dimension - 1)]
 */
struct Coord4D : Base4D<Coord4D> {
    static constexpr TraitType trait_mask = USE_ALIGN_METHODS_TRAIT;

    Coord4D getEndWithShapeAdded(const Shape4D& shape) const {
        return {{.n = getEndCoordinate(n, shape.n),
                 .c = getEndCoordinate(c, shape.c),
                 .h = getEndCoordinate(h, shape.h),
                 .w = getEndCoordinate(w, shape.w)}};
    }
    Shape4D getShapeUpToEnd(const Coord4D& end) const {
        return {{.n = getLengthFromStartToEnd(n, end.n),
                 .c = getLengthFromStartToEnd(c, end.c),
                 .h = getLengthFromStartToEnd(h, end.h),
                 .w = getLengthFromStartToEnd(w, end.w)}};
    }
    bool isValidIn(const Shape4D& shape) const { return w < shape.w && h < shape.h && c < shape.c && n < shape.n; }
};

inline std::ostream& operator<<(std::ostream& s, const Coord4D& coord) {
    s << "(n:" << coord.n << " c:" << coord.c << " h:" << coord.h << " w:" << coord.w << ')';
    return s;
}

typedef struct NodeInfo_ {
    explicit NodeInfo_(NODE_ID_T              id,
                       std::string            name,
                       const nn_ir::NNIR&     graph,
                       std::vector<EDGE_ID_T> in_edge_ids  = std::vector<EDGE_ID_T>(),
                       std::vector<EDGE_ID_T> out_edge_ids = std::vector<EDGE_ID_T>())
        : id(id), name(std::move(name)), graph(graph), in_edge_ids(std::move(in_edge_ids)),
          out_edge_ids(std::move(out_edge_ids)) {}

    NODE_ID_T              id;
    std::string            name;
    const nn_ir::NNIR&     graph;
    std::vector<EDGE_ID_T> in_edge_ids;
    std::vector<EDGE_ID_T> out_edge_ids;
} NodeInfo;

typedef struct BlobInfo_ {
    explicit BlobInfo_(BLOB_ID_T                         id,
                       std::string                       name,
                       const nn_ir::NNIR&                graph,
                       BlobType                          blob_type,
                       DataType                          data_type,
                       QuantType                         quant_type,
                       ShapeType                         shape_type,
                       Shape4D                           dim,
                       Shape4D                           alignment_unit,
                       BIT_WIDTH_T                       bit_width,
                       std::pair<NODE_ID_T, NODE_ID_T>   liveness,
                       int32_t                           zero_point,
                       bool                              compress,
                       const std::vector<FRAC_LENGTH_T>& frac_len)
        : id(id), name(std::move(name)), graph(graph), blob_type(blob_type), data_type(data_type),
          quant_type(quant_type), shape_type(shape_type), dim(dim), alignment_unit(alignment_unit),
          bit_width(bit_width), liveness(std::move(liveness)), zero_point(zero_point), compress(compress),
          frac_len(frac_len) {}

    BLOB_ID_T                       id;
    std::string                     name;
    const nn_ir::NNIR&              graph;
    BlobType                        blob_type;
    DataType                        data_type;
    QuantType                       quant_type;
    ShapeType                       shape_type;
    Shape4D                         dim;
    Shape4D                         alignment_unit;
    BIT_WIDTH_T                     bit_width;
    std::pair<NODE_ID_T, NODE_ID_T> liveness;
    int32_t                         zero_point;
    bool                            compress;
    std::vector<FRAC_LENGTH_T>      frac_len;
} BlobInfo;

typedef struct EdgeInfo_ {
    explicit EdgeInfo_(
        EDGE_ID_T id, std::string name, const nn_ir::NNIR& graph, NODE_ID_T in_node_id, NODE_ID_T out_node_id)
        : id(id), name(std::move(name)), graph(graph), in_node_id(in_node_id), out_node_id(out_node_id) {}

    EDGE_ID_T   id;
    std::string name;
    const NNIR& graph;
    NODE_ID_T   in_node_id;
    NODE_ID_T   out_node_id;
} EdgeInfo;

enum class MemoryType {
    DRAM = 0,
    SRAM = 1,
    FIFO = 2,
};

inline std::ostream& operator<<(std::ostream& s, MemoryType mem_type) {
    switch (mem_type) {
#define ENUM_STR(x)     \
    case MemoryType::x: \
        s << #x;        \
        break;
        ENUM_STR(DRAM)
        ENUM_STR(SRAM)
        ENUM_STR(FIFO)
#undef ENUM_STR
        default:
            Log::IR::E() << "Invalid MemoryType " << static_cast<int>(mem_type);
            break;
    }
    return s;
}

enum class MemoryDataType {
    PSUM            = 0,
    IFM             = 1,
    KERNEL          = 2,
    OFM             = 3,
    INSTR           = 4,
    MISC            = 5,
    INTERMEDIATE_FM = 6,
    SHARED_IM       = 7,
    CONSTANT        = 8,
    LUT             = 9,
    INVALID         = 10, // for initialization only
};

//  Just 'LITTLE_ENDIAN' is reserved word.
enum class PixelByteOrder {
    LITTLE_ENDIAN_ORDER       = 0,
    LITTLE_ENDIAN_SPLIT_ORDER = 1,
    INVALID_ORDER             = 2,
};

typedef struct DataLayout_ {
    Shape4D total_dim = {{.n = 0, .c = 0, .h = 0, .w = 0}}; // Dimensions including padding
    Pad4    padding   = {0, 0, 0, 0};                       // Padding around the featuremap
    Shape4D gap       = {{.n = 0, .c = 0, .h = 0, .w = 0}}; // Gaps between elements in respective directions in bytes
                                                      // gap.n specifies reserved space following the whole 3D tensor.
    Shape4D        cell_unit  = {{.n = 0, .c = 0, .h = 0, .w = 0}}; // Invalid value by default, helps to catch errors
    uint32_t       bpp        = 1;                                  // Bytes per pixel
    PixelByteOrder byte_order = PixelByteOrder::INVALID_ORDER;      // Byte order in one pixel

    DataLayout_() = default;
    // Some shorthand constructors for trivial layouts
    DataLayout_(const Shape4D& cell_size, uint32_t bpp, PixelByteOrder byte_order)
        : cell_unit(cell_size), bpp(bpp), byte_order(byte_order) {}
    DataLayout_(const Shape4D& total_dim, const Shape4D& cell_size, uint32_t bpp, PixelByteOrder byte_order)
        : total_dim(total_dim), cell_unit(cell_size), bpp(bpp), byte_order(byte_order) {}
    DataLayout_(const Shape4D& dim, const Pad4& pad, const Shape4D& cell_size, uint32_t bpp, PixelByteOrder byte_order)
        : total_dim(dim + pad), padding(pad), cell_unit(cell_size), bpp(bpp), byte_order(byte_order) {}
    DataLayout_(const Shape4D& total_dim,
                const Shape4D& blob_dim,
                const Shape4D& cell_size,
                uint32_t       bpp,
                PixelByteOrder byte_order)
        : total_dim(total_dim), cell_unit(cell_size), bpp(bpp), byte_order(byte_order) {
        padding.r = total_dim.w - blob_dim.w;
        padding.b = total_dim.h - blob_dim.h;
    }

    bool isRaster() const { return cell_unit.n == 1 && cell_unit.c == 1 && cell_unit.h == 1 && cell_unit.w == 1; }

    void extendAtBottomRight(uint32_t pad_r, uint32_t pad_b) {
        if (pad_r > padding.r) {
            total_dim.w += pad_r - padding.r;
            padding.r = pad_r;
        }
        if (pad_b > padding.b) {
            total_dim.h += pad_b - padding.b;
            padding.b = pad_b;
        }
    }

    void alignUpTo(const nn_ir::Shape4D& alignment_unit) {
        const nn_ir::Shape4D aligned_dim = total_dim.getAlignedUpBy(alignment_unit);

        // Extend bottom-right padding if alignment_unit requests so
        extendAtBottomRight(aligned_dim.w - total_dim.w, aligned_dim.h - total_dim.h);
        // And apply C and N alignment
        total_dim.c = aligned_dim.c;
        total_dim.n = aligned_dim.n;
    }

    // A packed layout has no gaps in between pixels. The only gap that is allowed to
    // be present is N, because it's a reserved space after the whole tensor.
    // Packed layouts can be Reshape'd on the fly. Network inputs and outputs are also
    // expected to be packed since the driver doesn't want to deal with such
    // HW-dependent requirements.
    bool isPacked() const {
        return padding.l == 0 && padding.r == 0 && padding.t == 0 && padding.b == 0 && gap.w == 0 && gap.h == 0 &&
               gap.c == 0;
    }

    uint32_t calcColStride() const;
    uint32_t calcRowStride() const;
    uint32_t calcChannelStride() const;
    // INT16 featuremaps are collated into two 8-bit CxHxW fragments, LSB fragment followed by MSB
    // Therefore byte_stride in bytes is equal in fact to the size of a single 8-bit half.
    // gap.n still doesn't apply here because it follows the complete data, both halves.
    // Again this is necessary in order to be able to represent both halves as a single 8-bit tensor
    // with (c * 2) channels for DMA transfers.
    uint32_t calcByteStride() const { return calcChannelStride() * total_dim.c; }

    MEMORY_SIZE_T calcSizeInBytes(nn_type::BlobUnitSize blob_unit_size) const;
} DataLayout;

inline std::ostream& operator<<(std::ostream& s, const DataLayout& l) {
    s << "padding:" << l.padding << " total_dim:" << l.total_dim << " gap:" << l.gap << " cell_unit:" << l.cell_unit;
    return s;
}

typedef struct MemoryInfo_ {
    MemoryType      memory_type;
    MemoryDataType  data_type;
    MEMORY_ID_T     mem_id;
    MEMORY_OFFSET_T addr       = INVALID_OFFSET;
    MEMORY_SIZE_T   size       = INVALID_SIZE;
    MEMORY_SIZE_T   msb_stride = 0; // An allocation is contiguous by default
    DataLayout      layout;
} MemoryInfo;

enum class SyncType {
    NONE   = 0,
    LOCAL  = 1,
    REMOTE = 2,
};

enum class SigType {
    NONE     = 0,
    SIG_WAIT = 1,
    SIG_BAR  = 2,
    SIG_SEND = 3,
};

/**
 * @brief Describes 4D position of a tile relative to other tiles in the layout
 * @example {n:0, c:2, h:3, w:1}  would mean "first tile-section in batch, 3rd in depth, 4th from top, 2nd from left"
 */
struct TilePosition : Base4D<TilePosition> {
    static constexpr TraitType trait_mask = USE_NO_EXTRA_METHODS_TRAIT;

    bool isValidIn(const TileNumbers& tile_numbers) const {
        return n < tile_numbers.n && c < tile_numbers.c && h < tile_numbers.h && w < tile_numbers.w;
    }

    /**
     * @brief Returns number (id) of tile in the layout by its 4D location in layout
     * TODO(a.puschin): get rid of this method, refine client code to use Position itself
     */
    uint32_t getIdIn(const TileNumbers& tile_numbers) const {
        Log::IR::E_IF(!this->isValidIn(tile_numbers)) << "Position requested is out of index boundaries";
        return tile_numbers.w * tile_numbers.h * tile_numbers.c * this->n + // number of full-instance tile-packs
               tile_numbers.w * tile_numbers.h * this->c +                  // number of full-channel tile-packs
               tile_numbers.w * this->h +                                   // number of full-row tile-packs
               this->w;                                                     // number of tiles in the last row
    }
};

inline std::ostream& operator<<(std::ostream& s, const TilePosition& pos) {
    s << "(n:" << pos.n << " c:" << pos.c << " h:" << pos.h << " w:" << pos.w << ')';
    return s;
}

typedef struct TileInfo_ {
    int32_t      node_id; // TODO(a.puschin): consider changing it into blob_id as blobs are tiled, not nodes
    TilePosition position;
    Coord4D      first_value_coord;
} TileInfo;

inline std::ostream& operator<<(std::ostream& s, const TileInfo& info) {
    s << "TileInfo for Node #" << info.node_id << ": {position=" << info.position
      << ", first_value_coord=" << info.first_value_coord << "}";
    return s;
}

/**
 * @brief Describes position of a tile relative to other tiles in the layout, along one dimension (axis) only
 */
enum class TileType {
    NONE  = 0, /// unknown type, error
    FIRST = 1, /// tile adjacent to the "front" boundary of the featuremap along this dimension
    MID   = 2, /// tile inside featuremap, doesn't touch boundaries along this dimension
    LAST  = 3, /// tile adjacent to the "back" boundary of the featuremap along this dimension
    FULL  = 4, /// the only-one tile that occupies whole featuremap along this dimension
};

union NNNodeUnionType {
    nn_ir::PoolType pool_type;
};

BIT_WIDTH_T getBitWidthByType(DataType type);
bool        isSignedByType(DataType type);

inline uint32_t getSizeInBytesByType(DataType type) { return getBitWidthByType(type) / CHAR_BIT; }

void printMemInfo(const std::vector<MemoryInfo>& mem_info);

// clang-format off

inline bool isFloatDataType(DataType data_type) {
    return data_type >= DataType::FloatDataType && data_type <= DataType::LastFloatDataType;
}

inline bool isIntDataType(DataType data_type) {
    return data_type >= DataType::IntDataType && data_type <= DataType::LastIntDataType;
}

template <typename DType>
inline DataType getDataType() {
    static_assert(std::is_same<DType, uint8_t>::value  ||
                  std::is_same<DType, int8_t>::value   ||
                  std::is_same<DType, uint16_t>::value ||
                  std::is_same<DType, int16_t>::value  ||
                  std::is_same<DType, uint32_t>::value ||
                  std::is_same<DType, int32_t>::value  ||
                  std::is_same<DType, int64_t>::value  ||
                  std::is_same<DType, float>::value    ||
                  std::is_same<DType, double>::value   ||
                  std::is_same<DType, float16>::value,
                  "This function does not support the specified type.");
    return DataType::NONE;
}

// clang-format on

template <>
inline DataType getDataType<uint8_t>() {
    return DataType::UINT8;
}

template <>
inline DataType getDataType<int8_t>() {
    return DataType::INT8;
}

template <>
inline DataType getDataType<uint16_t>() {
    return DataType::UINT16;
}

template <>
inline DataType getDataType<int16_t>() {
    return DataType::INT16;
}

template <>
inline DataType getDataType<uint32_t>() {
    return DataType::UINT32;
}

template <>
inline DataType getDataType<int32_t>() {
    return DataType::INT32;
}

template <>
inline DataType getDataType<int64_t>() {
    return DataType::INT64;
}

template <>
inline DataType getDataType<float>() {
    return DataType::FLOAT32;
}

template <>
inline DataType getDataType<double>() {
    return DataType::FLOAT64;
}

template <>
inline DataType getDataType<float16>() {
    return DataType::FLOAT16;
}

///
/// end casting functions
///

/// @brief get string name by memory data type
inline const char* getMemNameByMemType(nn_ir::MemoryDataType mem_type) {
    switch (mem_type) {
        case nn_ir::MemoryDataType::IFM:
            return "IFM";
        case nn_ir::MemoryDataType::OFM:
            return "OFM";
        case nn_ir::MemoryDataType::PSUM:
            return "PSUM";
        case nn_ir::MemoryDataType::KERNEL:
            return "KERNEL";
        case nn_ir::MemoryDataType::CONSTANT:
            return "CONSTANT";
        case nn_ir::MemoryDataType::INSTR:
            return "INSTR";
        case nn_ir::MemoryDataType::MISC:
            return "MISC";
        case nn_ir::MemoryDataType::INTERMEDIATE_FM:
            return "INTERMEDIATE_FM";
        case nn_ir::MemoryDataType::SHARED_IM:
            return "SHARED_IM";
        case nn_ir::MemoryDataType::INVALID:
            Log::IR::E() << "MemoryDataType was not initialized";
        default:
            return "UNKNOWN";
    }
}

inline std::ostream& operator<<(std::ostream& os, MemoryDataType mem_type) {
    return os << getMemNameByMemType(mem_type);
}

/// @brief get string name by tiling scheme
inline const char* getTilingSchemeName(TilingScheme scheme) {
    switch (scheme) {
        case TilingScheme::NONE:
            return "NONE";
        case TilingScheme::WEIGHT:
            return "WEIGHT";
        case TilingScheme::IFM:
            return "IFM";
        case TilingScheme::IFM_WEIGHT:
            return "IFM_WEIGHT";
    }
    Log::IR::E() << "Unknown TilingScheme";
}

// TODO(i-veselov): completely remove Tiling scheme
// TilingScheme is deprecated ir structure. Please, don't use it.
inline nn_ir::TilingScheme getTilingScheme(const TileNumbers& num_tiles) {
    if (num_tiles.hasSpatialTilesOnly()) {
        return nn_ir::TilingScheme::IFM;
    } else if (num_tiles.hasChannelwiseTilesOnly()) {
        return nn_ir::TilingScheme::WEIGHT;
    } else if (num_tiles.hasSpatialTiles() && num_tiles.hasChannelwiseTiles()) {
        return nn_ir::TilingScheme::IFM_WEIGHT;
    } else {
        return nn_ir::TilingScheme::NONE;
    }
}

inline std::ostream& operator<<(std::ostream& os, TilingScheme scheme) { return os << getTilingSchemeName(scheme); }

inline std::ostream& operator<<(std::ostream& os, const MemoryInfo& mem_info) {
    return os << "addr = " << mem_info.addr << " (0x" << std::hex << mem_info.addr << std::dec
              << "), size = " << mem_info.size << ", data_type = " << mem_info.data_type
              << ", memory_type = " << mem_info.memory_type;
}

using NNIR_Node_Param_ = std::variant<nn_ir::Shape2D, nn_ir::Shape4D, nn_ir::Pad4, nn_ir::TileInfo, nn_ir::TileNumbers>;
using NNIR_Node_Config_Type_ = std::variant<nn_ir::InputType,
                                            nn_ir::PadCalcType,
                                            nn_ir::PoolType,
                                            nn_ir::EltwiseType,
                                            nn_ir::ActivationType,
                                            nn_ir::PriorboxType,
                                            nn_ir::GlobalConcatAxis,
                                            nn_ir::GlobalConcatType,
                                            nn_ir::PartitionMode,
                                            nn_ir::TilingScheme,
                                            nn_ir::Axis,
                                            nn_ir::SigType,
                                            nn_ir::SyncType,
                                            nn_ir::BlobType,
                                            nn_ir::QuantType,
                                            nn_ir::ShapeType,
                                            nn_ir::DataType,
                                            nn_ir::NodeExecutionStepType,
                                            nn_ir::EdgeExecutionStepType,
                                            nn_ir::MemoryType,
                                            nn_ir::MemoryDataType,
                                            nn_ir::PixelByteOrder>;
using IR_Node_Param_         = std::variant<const IR::Type::Dim2*,
                                    const IR::Type::Dim4*,
                                    const IR::Type::Pad4*,
                                    const IR::Type::TilePosition*,
                                    const IR::Type::TileNumbers*>;
using IR_Node_Config_Type_   = std::variant<IR::NNNode::InputType,
                                          IR::NNNode::PadCalculation,
                                          IR::NNNode::PoolType,
                                          IR::NNNode::EltwiseType,
                                          IR::NNNode::ActivationType,
                                          IR::GlobalNode::PartitionModeType,
                                          IR::GlobalNode::SyncType,
                                          IR::GlobalNode::SigType,
                                          IR::GlobalNode::GlobalConcatType,
                                          IR::GlobalNode::GlobalConcatAxis,
                                          IR::Type::PriorboxType,
                                          IR::Type::TilingSchemeType,
                                          IR::Type::TilingDirectionType,
                                          IR::Type::BlobType,
                                          IR::Type::QuantType,
                                          IR::Type::ShapeType,
                                          IR::Type::DataType,
                                          IR::TargetHardware::Type::NodeExecutionType,
                                          IR::TargetHardware::Type::EdgeExecutionType,
                                          IR::TargetHardware::Type::MemoryType,
                                          IR::TargetHardware::Type::MemoryDataType,
                                          IR::TargetHardware::Type::PixelByteOrder>;

template <class... Ts>
struct overloaded : Ts... {
    using Ts::operator()...;
};
template <class... Ts>
overloaded(Ts...)->overloaded<Ts...>;
NNIR_Node_Param_       parseParam(IR_Node_Param_ type);
NNIR_Node_Config_Type_ parseConfigType(IR_Node_Config_Type_& type);
IR_Node_Config_Type_   parseConfigType(NNIR_Node_Config_Type_& type);

struct LUTBlobs {
    Blob* expLUTBlob     = nullptr;
    Blob* softmaxLUTBlob = nullptr;

    bool isValid() const { return expLUTBlob != nullptr && softmaxLUTBlob != nullptr; }
};

} // namespace nn_ir
} // namespace nn_compiler

/*
 * @brief: Implements default filter iterator traits
 */
template <typename ValueT>
struct DefaultFilterIteratorTraits {
    using ValueType = typename std::decay<ValueT>::type;
    template <typename From>
    static bool isKindOf(const From& val) {
        return isa<ValueType>(val);
    }
    template <typename From>
    static auto& cast(From&& val) {
        return ::cast<ValueType>(std::forward<From>(val));
    }
    template <typename From, typename PredT>
    static bool isKindOf(const From& val, const PredT& pred) {
        using FromPureT  = typename std::remove_cv<typename std::remove_reference<From>::type>::type;
        using FromValueT = typename std::remove_cv<typename std::remove_reference<ValueType>::type>::type;
        if constexpr (std::is_same_v<FromPureT, FromValueT>) {
            return pred(val);
        } else {
            return isa<ValueType>(val) && pred(cast(val));
        }
    }
};
