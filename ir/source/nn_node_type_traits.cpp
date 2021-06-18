/*
 * Copyright (C) 2019 Samsung Electronics Co. LTD
 *
 * This software is proprietary of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#include "ir/include/nn_node_type_traits.hpp"

#include <optional>

namespace nn_compiler {
namespace nn_ir {

static ShiftNode* getShiftNodeImpl(Node& node) {
    switch (node.getNodeType()) {
#define SHIFTABLE_NODE(ENUM_TYPE, C_TYPE) \
    case ENUM_TYPE:                       \
        return static_cast<C_TYPE&>(node).getShiftNode();
#include "ir/source/private/shiftable_nodes.inc"
        default:
            return nullptr;
    }
}

static void setShiftNodeImpl(Node& node, std::unique_ptr<ShiftNode> shift_node) {
    switch (node.getNodeType()) {
#define SHIFTABLE_NODE(ENUM_TYPE, C_TYPE)                               \
    case ENUM_TYPE:                                                     \
        static_cast<C_TYPE&>(node).setShiftNode(std::move(shift_node)); \
        break;
#include "ir/source/private/shiftable_nodes.inc"
        default:
            Log::IR::I() << "setShiftNode() => unsupported node type!";
            return;
    }
}

static const ActivationNode* getActivationNodeImpl(const Node& node) {
    switch (node.getNodeType()) {
#define ACTIVATABLE_NODE(ENUM_TYPE, C_TYPE) \
    case ENUM_TYPE:                         \
        return static_cast<const C_TYPE&>(node).getActivationNode();
#include "ir/source/private/activatable_nodes.inc"
        default:
            return nullptr;
    }
}

static void setActivationNodeImpl(Node& node, std::unique_ptr<ActivationNode> act_node) {
    switch (node.getNodeType()) {
#define ACTIVATABLE_NODE(ENUM_TYPE, C_TYPE)                                \
    case ENUM_TYPE:                                                        \
        static_cast<C_TYPE&>(node).setActivationNode(std::move(act_node)); \
        break;
#include "ir/source/private/activatable_nodes.inc"
        default:
            Log::IR::I() << "setActivationNode() => unsupported node type!";
            return;
    }
}

static BLOB_ID_T getKernelBlobIdImpl(const Node& node) {
    switch (node.getNodeType()) {
#define NODE_WITH_KERNEL_DATA(ENUM_TYPE, C_TYPE) \
    case ENUM_TYPE:                              \
        return static_cast<const C_TYPE&>(node).getKernelBlobId();
#include "ir/source/private/nodes_with_kernel_data.inc"
        default:
            return INVALID_ID;
    }
}

static void setKernelBlobIdImpl(Node& node, BLOB_ID_T blob_id) {
    switch (node.getNodeType()) {
#define NODE_WITH_KERNEL_DATA(ENUM_TYPE, C_TYPE)             \
    case ENUM_TYPE:                                          \
        static_cast<C_TYPE&>(node).setKernelBlobId(blob_id); \
        break;
#include "ir/source/private/nodes_with_kernel_data.inc"
        default:
            Log::IR::I() << "setKernelBlobId() => unsupported node type!";
            return;
    }
}

static BLOB_ID_T getBiasBlobIdImpl(const Node& node) {
    switch (node.getNodeType()) {
#define NODE_WITH_KERNEL_DATA(ENUM_TYPE, C_TYPE) \
    case ENUM_TYPE:                              \
        return static_cast<const C_TYPE&>(node).getBiasBlobId();
#include "ir/source/private/nodes_with_kernel_data.inc"
        default:
            return INVALID_ID;
    }
}

static void setBiasBlobIdImpl(Node& node, BLOB_ID_T blob_id) {
    switch (node.getNodeType()) {
#define NODE_WITH_KERNEL_DATA(ENUM_TYPE, C_TYPE)           \
    case ENUM_TYPE:                                        \
        static_cast<C_TYPE&>(node).setBiasBlobId(blob_id); \
        break;
#include "ir/source/private/nodes_with_kernel_data.inc"
        default:
            Log::IR::I() << "setBiasBlobId() => unsupported node type!";
            return;
    }
}

static std::optional<KernelNodeParameters> getKernelNodeParametersImpl(const Node& node) {
    switch (node.getNodeType()) {
#define KERNEL_NODE(ENUM_TYPE, C_TYPE) \
    case ENUM_TYPE:                    \
        return static_cast<const C_TYPE&>(node).getKernelNodeParameters();
#include "ir/source/private/kernel_nodes.inc"
        default:
            return std::nullopt;
    }
}

static void setKernelNodeParametersImpl(Node& node, const KernelNodeParameters& kernel_node_parameters) {
    switch (node.getNodeType()) {
#define KERNEL_NODE(ENUM_TYPE, C_TYPE)                                              \
    case ENUM_TYPE:                                                                 \
        static_cast<C_TYPE&>(node).setKernelNodeParameters(kernel_node_parameters); \
        break;
#include "ir/source/private/kernel_nodes.inc"
        default:
            Log::IR::I() << "setKernelNodeParameters() => unsupported node type!";
            return;
    }
}

static bool isChannelwiseImpl(const Node& node) {
    switch (node.getNodeType()) {
#define DEPTHWISE_BY_DEFAULT_NODE(ENUM_TYPE, C_TYPE) \
    case ENUM_TYPE:                                  \
        return true;
#include "ir/source/private/channelwise_by_default_nodes.inc"
        default:
            if (auto* kernel_blob = nn_ir::getKernelBlob(node)) {
                auto  kernel_blob_dim = kernel_blob->getShape();
                auto& data_edge       = cast<nn_ir::DataEdge>(node.getFirstInEdge());
                auto  ifm_blob_dim    = data_edge.getBlob()->getShape();
                return kernel_blob_dim.n == ifm_blob_dim.c && kernel_blob_dim.c == 1;
            }
            return false;
    }
}

ShiftNode* getShiftNode(Node& node) { return getShiftNodeImpl(node); }

void setShiftNode(Node& node, std::unique_ptr<ShiftNode> shift_node) { setShiftNodeImpl(node, std::move(shift_node)); }

const ActivationNode* getActivationNode(const Node& node) { return getActivationNodeImpl(node); }

void setActivationNode(Node& node, std::unique_ptr<ActivationNode> activation_node) {
    setActivationNodeImpl(node, std::move(activation_node));
}

Blob* getKernelBlob(const Node& node) {
    BLOB_ID_T blob_id = getKernelBlobIdImpl(node);
    return blob_id == INVALID_ID ? nullptr : node.getGraph().getBlob(blob_id);
}

LUTBlobs getLutBlobs(const Node& node) {
    LUTBlobs lut_blobs;
    if (auto* softmax_node = cast_if<nn_ir::SoftmaxNode>(node)) {
        lut_blobs.expLUTBlob     = softmax_node->getGraph().getBlob(softmax_node->getExpLUTBlobId());
        lut_blobs.softmaxLUTBlob = softmax_node->getGraph().getBlob(softmax_node->getSoftmaxLUTBlobId());
    }
    return lut_blobs;
}

void setKernelBlobId(Node& node, BLOB_ID_T blob_id) { setKernelBlobIdImpl(node, blob_id); }

Blob* getBiasBlob(const Node& node) {
    BLOB_ID_T blob_id = getBiasBlobIdImpl(node);
    return blob_id == INVALID_ID ? nullptr : node.getGraph().getBlob(blob_id);
}

void setBiasBlobId(Node& node, BLOB_ID_T blob_id) { setBiasBlobIdImpl(node, blob_id); }

/** @brief Check if the node is executable
 *
 * An Executable operation is the one that actually transforms the data, involving
 * some hardware operations. It has distinct input and output blob(s).
 */
bool isExecutableNode(const Node& node) {
    // ConcatNode is NNNode historically, but it's not executable, it's a memory hint
    return is_any_of<NNNode, HWNode, QNode>(node) && !isa<ConcatNode>(node);
}

/** @brief Check if the node is a memory allocator hint
 *
 * Memory allocator hints are special kind of nodes, not representing any actual
 * data transformations. They have a single underlying memory allocation, with inputs
 * and outputs somehow located inside that allocation. Consequently, all their inputs
 * and outputs share the same memory type and region.
 */
bool isMemoryHint(const Node& node) { return is_any_of<ConcatNode, VConcatNode, VSplitNode, GlobalNode>(node); }

Blob* getPregeneratedBlob(const Node& node) {
    if (auto priorbox = cast_if<PriorBoxNode>(node)) {
        return priorbox->getBlob();
    }
    return getKernelBlob(node);
}

bool isSyncOp(const Node& node) {
    return isa<nn_ir::GlobalNode>(node) && cast<nn_ir::GlobalNode>(node).getSyncType() != SyncType::NONE;
}

bool isBranchConcat(const Node& node) {
    if (auto gconcat = cast_if<nn_ir::GlobalConcatNode>(node)) {
        return gconcat->isBranchConcat();
    } else {
        return isa<nn_ir::ConcatNode>(node);
    }
}

bool isTileConcat(const Node& node) {
    if (auto gconcat = cast_if<nn_ir::GlobalConcatNode>(node)) {
        return gconcat->isTileConcat();
    }
    return false;
}

bool areInputsAlignedAt(const Node& node, uint32_t align) {
    for (const nn_ir::DataEdge& in : node.getInEdges<nn_ir::DataEdge>()) {
        // Last input of the Concat doesn't have to be aligned
        if ((in.getId() != node.getInEdgeIds().back()) && (in.getBlob()->getShape().c % align)) {
            return false;
        }
    }
    return true;
}

KernelNodeParameters getKernelNodeParameters(const Node& node) {
    auto kernel_params = getKernelNodeParametersImpl(node);

    // FIXME: for now some users suggest that if node has no kernel parameters then empty
    //        parameters will be returned. Maybe it's better to replace
    //        with error reporting in this case
    return kernel_params.has_value() ? *kernel_params : KernelNodeParameters();
}

void setKernelNodeParameters(Node& node, const KernelNodeParameters& kernel_node_parameters) {
    setKernelNodeParametersImpl(node, kernel_node_parameters);
}

bool hasKernelParameters(const Node& node) { return getKernelNodeParametersImpl(node).has_value(); }

bool isChannelwise(const Node& node) { return isChannelwiseImpl(node); }

static bool hasRasterOfmBlob(const Node& node, bool reverse);

static bool hasRasterIfmBlob(const Node& node, bool reverse) {
    const Node* pNode = &node;

    // MemoryHint nodes inherit this property from OpNodes' IFM behind them.
    // Since we'are being asked about IFM, we are walking down the graph
    while (isMemoryHint(*pNode)) {
        pNode = pNode->getFirstSuccessorNode();
        if (!pNode) {
            // If we've hit an output, we need to go the other way and check predecessor's OFM
            // This prevents endless looping back and forth if our "node" is a MemHint and belongs
            // to both IFM and OFM regions, i. e. one of input edges is graph input and one of
            // output edges is graph output. This must never happen as IFM and OFM are
            // distinct DRAM regions for us, which must be isolated by a bulk copy operation
            Log::IR::E_IF(reverse) << "MemHint " << node << " between input and output";
            return hasRasterOfmBlob(node, true);
        }
    }

    // At this point we've reached an executable node
#define RASTER_IFM_NODE(TYPE, COND)      \
    if (auto N = cast_if<TYPE>(pNode)) { \
        (void)N;                         \
        return (COND);                   \
    }
#include "ir/source/private/raster_ifm_nodes.inc"
    return false;
}

static bool hasRasterOfmBlob(const Node& node, bool reverse) {
    const Node* pNode = &node;

    // MemoryHint nodes inherit this property from OpNodes' OFM behind them.
    // Since we'are being asked about OFM, we are walking down the graph
    while (isMemoryHint(*pNode)) {
        pNode = pNode->getFirstPredecessorNode();
        if (!pNode) {
            // If we've hit an input, we need to go the other way and check successor's IFM
            // This prevents endless looping back and forth if our "node" is a MemHint and belongs
            // to both IFM and OFM regions, i. e. one of input edges is graph input and one of
            // output edges is graph output. This must never happen as IFM and OFM are
            // distinct DRAM regions for us, which must be isolated by a bulk copy operation
            Log::IR::E_IF(reverse) << "MemHint " << node << " between input and output";
            return hasRasterIfmBlob(node, true);
        }
    }

    // At this point we've reached an executable node
#define RASTER_OFM_NODE(TYPE, COND)      \
    if (auto N = cast_if<TYPE>(pNode)) { \
        (void)N;                         \
        return (COND);                   \
    }
#include "ir/source/private/raster_ofm_nodes.inc"
    return false;
}

bool hasRasterIfmBlob(const Node& node) {
    // We know that there's an OpNode before VConcat and after VSplit, rely on that
    // for better performance.
    // VSplit and VConcat are MemHints, so all their edges share the same format
    if (isa<VSplitNode>(node)) {
        return hasRasterIfmBlob(*node.getFirstSuccessorNode());
    } else if (isa<VConcatNode>(node)) {
        return hasRasterOfmBlob(*node.getFirstPredecessorNode());
    } else {
        return hasRasterIfmBlob(node, false);
    }
}

bool hasRasterOfmBlob(const Node& node) {
    if (isa<VSplitNode>(node)) {
        return hasRasterIfmBlob(*node.getFirstSuccessorNode());
    } else if (isa<VConcatNode>(node)) {
        return hasRasterOfmBlob(*node.getFirstPredecessorNode());
    } else {
        return hasRasterOfmBlob(node, false);
    }
}

bool isInputBoundariesOverflowAllowed(const Node& node) {
    if (const auto pool_node = cast_if<nn_ir::PoolNode>(node);
        pool_node->getPoolType() == nn_ir::PoolType::AVERAGE &&
        pool_node->getPadCalcType() == nn_ir::PadCalcType::INCLUDE) {
        return true;
    }
    return false;
}

bool mayIfmTilesOverlap(const Node& node) {
    /** @brief
     *       |-|<---overlap
     * [A][B][C][D][E][F]...  <- IFM
     * [k][k][k]              <- dilated kernel
     *  ╎    [k][k][k]        <- dilated kernel is moved right on 'stride'
     *  ╎ ╭────╯    [k][k][k]
     *  ╎ ╎   ╭────────╯
     * [a][b][c][d]...
     *
     * If the first ofm tile is [a][b] and the second is [c][d],
     * then we need for the first  ofm tile take these IFM coords: [A][B][C][D][E]
     *              for the second ofm tile take these IFM coords: [E][F][G][H][I]
     * Both IFM tiles overlap in the [E] point
     * This happends when kernel size is greater than stride size.
     */

    const auto& kernel_params = nn_ir::getKernelNodeParameters(node);

    const auto& dilation_size       = kernel_params.getDilationSize();
    const auto& dilated_kernel_size = kernel_params.getKernelSize().getDilated(dilation_size);
    const auto& stride_size         = kernel_params.getStrideSize();

    return dilated_kernel_size > stride_size;
}

std::vector<const nn_ir::Blob*> getIfmBlobs(const Node& node) {
    std::vector<const nn_ir::Blob*> ifm_blobs;

    const auto& in_edges = node.getInEdges<nn_ir::DataEdge>();

    std::transform(in_edges.begin(), in_edges.end(), std::back_inserter(ifm_blobs), [](const auto& in_edge) {
        return in_edge.getBlob();
    });

    return ifm_blobs;
}

const nn_ir::Blob* getFirstOfmBlob(const Node& node) {
    return ::cast<nn_ir::DataEdge>(node.getFirstOutEdge()).getBlob();
}

nn_ir::Shape4D getFirstIfmShape(const Node& node) { return getIfmBlobs(node).front()->getShape(); }

nn_ir::Shape4D getFirstOfmShape(const Node& node) { return getFirstOfmBlob(node)->getShape(); }

bool isRemoteSync(const nn_ir::Node& node) {
    auto global_node = cast_if<nn_ir::GlobalNode>(node);
    return global_node && global_node->getSyncType() == nn_ir::SyncType::REMOTE;
}

bool isLocalSync(const nn_ir::Node& node) {
    auto global_node = cast_if<nn_ir::GlobalNode>(node);
    return global_node && global_node->getSyncType() == nn_ir::SyncType::LOCAL;
}

bool loadsInExecStep(const nn_ir::Node& node) { return isSyncOp(node) || node.hasIfmInDram(); }

bool storesInExecStep(const nn_ir::Node& node) { return isSyncOp(node) || node.hasOfmInDram(); }

NodeExecSteps getExecutionSteps(const nn_ir::Node& node) {
    NodeExecSteps steps;
    if (!loadsInExecStep(node)) {
        // IFM does not need to be loaded for some nodes
        steps.loads.reserve(node.getPredecessorsNum() + 1);
        for (const auto& data_edge : node.getInEdges<nn_ir::DataEdge>()) {
            steps.loads.push_back(&data_edge.getStep(nn_ir::EdgeExecutionStepType::LOAD_START));
        }
        // Sram-to-Dram ops will end up here, but they don't need NODE_DATA_LOAD_START
        if (!node.hasOfmInDram()) {
            steps.loads.push_back(&node.getStep(nn_ir::NodeExecutionStepType::NODE_DATA_LOAD_START));
        }
    }

    steps.exec = &node.getStep(nn_ir::NodeExecutionStepType::EXEC_START);

    if (!storesInExecStep(node)) {
        steps.stores.reserve(node.getSuccessorsNum());
        for (const auto& data_edge : node.getOutEdges<nn_ir::DataEdge>()) {
            steps.stores.push_back(&data_edge.getStep(nn_ir::EdgeExecutionStepType::STORE_START));
        }
    }

    return steps;
}

} // namespace nn_ir
} // namespace nn_compiler
