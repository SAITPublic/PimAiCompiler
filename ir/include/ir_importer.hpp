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

#include "common/include/types.hpp"
#include "ir/include/nn_ir.hpp"

namespace nn_compiler {

class IRImporter {
 public:
    /**
     * @brief.      Constructor of IRImporter.
     * @details.    This function constructs IRImporter
     * @param[in].
     * @param[out].
     * @returns.
     */
    IRImporter() = default;

    IRImporter(const IRImporter&) = delete;
    IRImporter(IRImporter&&)      = delete;
    IRImporter& operator=(const IRImporter&) = delete;
    IRImporter& operator=(IRImporter&&) = delete;

    /**
     * @brief.      Create nn_ir::NNIR class from IR file.
     * @details.    This function parses IR file (flatbuffer) and
     *              instantiate an object of nn_ir::NNIR class
     * @param[in].  file_path Input file path
     * @param[out]. graphs nn_ir::NNIR graph list
     * @returns.    return code
     */
    RetVal getNNIRFromFile(const std::string& file_path, std::vector<std::unique_ptr<nn_ir::NNIR>>& graphs);
    RetVal buildNNIRFromData(const char* data, std::vector<std::unique_ptr<nn_ir::NNIR>>& graphs);
}; // class IRImporter
} // namespace nn_compiler
