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

#include "ir/include/blob.hpp"
#include "ir/include/generated/ir_generated.h"
#include "ir/include/nn_ir.hpp"

namespace nn_compiler {

class IRBlobBuilder {
 public:
    /**
     * @brief.      Constructor of IRBlobBuilder.
     * @details.    This function constructs IRBlobBuilder
     * @param[in].
     * @param[out].
     * @returns.
     */
    IRBlobBuilder() = default;

    IRBlobBuilder(const IRBlobBuilder&) = delete;
    IRBlobBuilder(IRBlobBuilder&&)      = delete;
    IRBlobBuilder& operator=(const IRBlobBuilder&) = delete;
    IRBlobBuilder& operator=(IRBlobBuilder&&) = delete;

    /**
     * @brief.      getOrCreateBlob
     * @details.    This function retrieves Blob instance corresponding to the IR description
     * @param[in].  blob A blob of flatbuffer
     * @param[out].
     * @returns.    Blob pointer
     */
    nn_ir::Blob* getOrCreateBlob(const IR::Blob* ir_blob, nn_ir::NNIR& graph);

 private:
    /**
     * @brief.      createBlobFromIR
     * @details.    This function creates Blob instance from IR
     * @param[in].  blob A blob of flatbuffer
     * @param[out].
     * @returns.    return code
     */
    std::unique_ptr<nn_ir::Blob> createBlob(const IR::Blob* blob, nn_ir::NNIR& graph);
}; // class IRBlobBuilder
} // namespace nn_compiler
