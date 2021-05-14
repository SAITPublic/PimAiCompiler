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
 * @file.    ir_importer.hpp
 * @brief.   This is IRImporter class
 * @details. This header defines IRImporter class.
 * @version. 0.1.
 */

#pragma once

#include "ir/edge.hpp"
#include "ir/generated/ir_generated.h"
#include "ir/ir_types.hpp"
#include "ir/nn_ir.hpp"

namespace nn_compiler {

class IRGraphBuilder {
 public:
    /**
     * @brief.      Constructor of IRGraphBuilder.
     * @details.    This function constructs IRGraphBuilder
     * @param[in].
     * @param[out].
     * @returns.
     */
    IRGraphBuilder() = default;

    IRGraphBuilder(const IRGraphBuilder&) = delete;
    IRGraphBuilder(IRGraphBuilder&&)      = delete;
    IRGraphBuilder& operator=(const IRGraphBuilder&) = delete;
    IRGraphBuilder& operator=(IRGraphBuilder&&) = delete;

    /**
     * @brief.      createNodeFromIR
     * @details.    This function creates Node instance from IR
     * @param[in].  node A node of flatbuffer
     * @param[out].
     * @returns.    return code
     */
    std::unique_ptr<nn_ir::Node> createNode(const IR::Node* node, nn_ir::NNIR& graph);

    /**
     * @brief.      createEdgeFromIR
     * @details.    This function creates Edge instance from IR
     * @param[in].  edge A edge of flatbuffer
     * @param[out].
     * @returns.    return code
     */
    std::unique_ptr<nn_ir::Edge> createEdge(const IR::Edge* edge, nn_ir::NNIR& graph);

 private:
    std::map<BLOB_ID_T, std::vector<nn_ir::MemoryInfo>> blob_mem_infos_;
}; // class IRGraphBuilder
} // namespace nn_compiler
