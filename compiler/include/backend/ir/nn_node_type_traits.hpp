/*
 * Copyright (C) 2019 Samsung Electronics Co. LTD
 *
 * This software is proprietary of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#pragma once

#include "ir/ir_includes.hpp"
#include "ir/kernel_node_parameters.hpp"
#include "ir/nn_ir.hpp"

namespace nn_compiler::nn_ir {

ShiftNode* getShiftNode(Node& node);
void       setShiftNode(Node& node, std::unique_ptr<ShiftNode> shift_node);

inline const ShiftNode* getShiftNode(const Node& node) { return getShiftNode(const_cast<Node&>(node)); }

const ActivationNode* getActivationNode(const Node& node);
void                  setActivationNode(Node& node, std::unique_ptr<ActivationNode> activation_node);

Blob* getKernelBlob(const Node& node);
// TODO(r.rusyaev): when IDs are replaced with pointers this function will have to be replaced with setKernelBlob
void setKernelBlobId(Node& node, BLOB_ID_T blob_id);

LUTBlobs getLutBlobs(const Node& node);

Blob* getBiasBlob(const Node& node);
// TODO(r.rusyaev): when IDs are replaced with pointers this function will have to be replaced with setKernelBlob
void setBiasBlobId(Node& node, BLOB_ID_T blob_id);

bool isSyncOp(const Node& node);

// Pregenerated blobs include raster blob for PriorBoxNode and kernel blob for other nodes
Blob* getPregeneratedBlob(const Node& node);

// This node is not virtual, it is executed on hardware
//
bool isExecutableNode(const Node& node);

// This node is virtual and serves as a memory allocator hint to do something special
// (like split / concat)
bool isMemoryHint(const Node& node);

// Whether a Node requires raster format on input or output
// Work correctly only after IntrinsicMappingPass has been executed!
bool hasRasterIfmBlob(const Node& node);
bool hasRasterOfmBlob(const Node& node);

// TODO(anyone): these are here only because Concat can be expressed by two unrelated classes:
// ConcatNode and GlobalConcatNode. Fix that.
// See https://github.sec.samsung.net/SAIT-NPU-Compiler/NPUCompiler/issues/2785
bool isBranchConcat(const Node& node);
bool isTileConcat(const Node& node);
bool areInputsAlignedAt(const Node& node, uint32_t align);

KernelNodeParameters getKernelNodeParameters(const Node& node);
void                 setKernelNodeParameters(Node& node, const KernelNodeParameters& kernel_node_parameters);
bool                 hasKernelParameters(const Node& node);

bool isChannelwise(const Node& node);

// TODO(r.rusyaev): this function must be removed from traits interface
bool isInputBoundariesOverflowAllowed(const Node& node);

// TODO(r.rusyaev): this function must be removed from traits interface
bool mayIfmTilesOverlap(const Node& node);

std::vector<const nn_ir::Blob*> getIfmBlobs(const Node& node);

const nn_ir::Blob* getFirstOfmBlob(const Node& node);

nn_ir::Shape4D getFirstIfmShape(const Node& node);

nn_ir::Shape4D getFirstOfmShape(const Node& node);

// Is node for synchronization between NPU and DSP?
bool isRemoteSync(const nn_ir::Node& node);

// Is node for synchronization between NPU cores?
bool isLocalSync(const nn_ir::Node& node);

bool loadsInExecStep(const nn_ir::Node& node);
bool storesInExecStep(const nn_ir::Node& node);

// Helper struct for getExecutionSteps trait
struct NodeExecSteps {
    std::vector<const nn_ir::ExecutionStep*> loads;
    const nn_ir::ExecutionStep*              exec;
    std::vector<const nn_ir::ExecutionStep*> stores;
};

NodeExecSteps getExecutionSteps(const nn_ir::Node& node);

} // namespace nn_compiler::nn_ir
