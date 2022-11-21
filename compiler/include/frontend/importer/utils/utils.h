/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system, or translated into any human or computer language in any form by any
 * means, electronic, mechanical, manual or otherwise, or disclosed to third parties without the express written
 * permission of Samsung Electronics. (Use of the Software is restricted to non-commercial, personal or academic,
 * research purpose only)
 */

#pragma once

#include <torch/csrc/jit/ir/ir.h>
#include <torch/script.h>

namespace nn_compiler
{
namespace frontend
{
/**
 * @brief Get and Print All Nodes of torchScript model
 */
class TorchscriptPrinter
{
   public:
    TorchscriptPrinter() = default;

    int printGraphRecursive(std::string filename);

    void printNodeRecursive(torch::jit::Node *node, size_t level);

    void printScriptModelRecursive(std::string filename);

    ~TorchscriptPrinter() = default;
};

}  // namespace frontend
}  // namespace nn_compiler
