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
