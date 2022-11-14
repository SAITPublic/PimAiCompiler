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

namespace nn_compiler
{
namespace frontend
{
/***@Detail:
 *  There is no need to keep prim::SetAttr in our graph, it always comes out afer a prim::Variabl layer,
 *  and update its value by a computation result.
 *  the connection for prim::SetAttr is always like:
 *
 *       prim::Variable    a computation layer (e.g. aten::ge)
 *          |       \        /
 *          |      prim::SetAttr
 *          |
 *     prim::GetAttr
 *          |
 *
 *  The prim::Variable which is updated by prim::SetAttr is never used before, so we can remove
 *  prim::SetAttr and prim::Variable directly, change the structure to:
 *
 *         a computation layer (e.g. aten::ge)
 *                  |
 *            prim::GetAttr
 *                  |
 *
 *  Some prim::GetAttr may get attr from a computation layer rather than prim::Variable, so we
 *  remove them in later pass together.
 *
 ***/
class RemoveSetAttrLayers : public Pass
{
   public:
    RemoveSetAttrLayers();

    bool fitCondition(std::unique_ptr<nn_compiler::ir::NNModel>& model);

    void run(std::unique_ptr<nn_compiler::ir::NNModel>& model);

    ~RemoveSetAttrLayers() = default;

   private:
    std::vector<std::shared_ptr<nn_compiler::ir::NNLayer>> remove_layers_;
};

}  // namespace frontend
}  // namespace nn_compiler
