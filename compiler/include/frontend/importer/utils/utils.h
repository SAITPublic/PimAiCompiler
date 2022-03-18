#pragma once

#include <iostream>
#include <ostream>
#include <string>
#include <vector>

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
