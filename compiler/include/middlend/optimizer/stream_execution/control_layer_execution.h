/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system, or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 * (Use of the Software is restricted to non-commercial, personal or academic, research purpose only)
 */

#pragma once

#include "common/pass.hpp"
#include "ir/include/utils/graph_util.h"

namespace nn_compiler
{
namespace middlend
{
/** @Details:
 *  Process and help prim::If/prim::EndIf/prim::Loop/PrimEndLoop to find their next layer in execution time.
 *  (1) For Prim::If layer, set where is the start layer of its then-net as well as else-net.
 *  (2) For Prim::EndIf layer, set which is the next layer after this sub-net (then-net or else-net).
 *  (3) For Prim::Loop layer, set where is the next layer after this loop.
 *  (4) For Prim::EndLoop layer, set where is the next layer after loop body runs, i.e., its corresponding Prim::Loop
 *      layer.
 **/
class ControlLayerExecution : public Pass
{
   public:
    ControlLayerExecution();

    bool fitCondition(std::unique_ptr<nn_compiler::ir::NNModel>& model);

    void run(std::unique_ptr<nn_compiler::ir::NNModel>& model);

    ~ControlLayerExecution() = default;

   private:
    std::vector<std::shared_ptr<nn_compiler::ir::NNLayer>> control_layers_;

};  // class ControlLayerExecution

}  // namespace middlend
}  // namespace nn_compiler
