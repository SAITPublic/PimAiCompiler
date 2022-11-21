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

#include "common/pass.hpp"

namespace nn_compiler
{
namespace frontend
{
/** @Details:
 *  If inputs of a prim::ListConstrcut are all constants, it is possible to do the work of prim::ListConstruct
 *  directly in frontend. The constructed value will be stored in a prim::Variable layer.
 **/
class ConstructList : public Pass
{
   public:
    ConstructList();

    bool fitCondition(std::unique_ptr<nn_compiler::ir::NNModel>& graph_model);

    void run(std::unique_ptr<nn_compiler::ir::NNModel>& graph_model);

    ~ConstructList() = default;

   private:
    std::vector<
        std::pair<std::shared_ptr<nn_compiler::ir::NNLayer>, std::vector<std::shared_ptr<nn_compiler::ir::DTensor>>>>
        process_layer_and_dtensor_;
};
}  // namespace frontend
}  // namespace nn_compiler
