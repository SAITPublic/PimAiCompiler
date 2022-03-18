#pragma once

#include "common/include/common.hpp"
#include "ir/include/layers/all_layers.h"
#include "ir/include/nn_model.h"

namespace nn_compiler
{
class Pass
{
   public:
    Pass() {}

    void add(std::shared_ptr<Pass> pass)
    {
        if (successor_) {
            successor_->add(pass);
        } else {
            successor_ = pass;
        }
    }

    virtual std::shared_ptr<Pass> getSuccessor() { return successor_; }

    virtual bool fitCondition(std::unique_ptr<nn_compiler::ir::NNModel>& model) { return true; }

    virtual void run(std::unique_ptr<nn_compiler::ir::NNModel>& model) { successor_->run(model); }

    virtual ~Pass() = default;

   protected:
    std::shared_ptr<Pass> successor_ = nullptr;
};

}  // namespace nn_compiler
