#pragma once

#include "ir/include/nn_ir.hpp"

#include <vector>

namespace nn_compiler {
namespace middlend {

class MiddlendDriver {
 public:
    MiddlendDriver() = default;

    void initialize();

    void importIR();

    void runPasses();

    void exportIR();

    void finalize();

 private:
    std::vector<std::unique_ptr<nn_compiler::nn_ir::NNIR>> graphs_;

}; // class MiddlendDriver

} // namespace middlend
} // namespace nn_compiler
