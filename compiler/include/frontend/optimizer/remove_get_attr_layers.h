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
/***@Detail:
 *  There is no need to keep prim::GetAttr in our graph, because the connection for prim::GetAttr
 *  is always like:
 *       prim::Variable (might be deleted by: remove_set_attr_layers)
 *            |
 *       prim::GetAttr
 *            |
 *    a computation op (e.g. aten::len)
 *            |
 *
 *  We can remove prim::GetAttr directly, change to:
 *      prim::Variable (might be deleted)
 *            |
 *    a computation op (e.g. aten::len)
 *            |
 ***/
class RemoveGetAttrLayers : public Pass
{
   public:
    RemoveGetAttrLayers();

    bool fitCondition(std::unique_ptr<nn_compiler::ir::NNModel>& model);

    void run(std::unique_ptr<nn_compiler::ir::NNModel>& model);

    ~RemoveGetAttrLayers() = default;

   private:
    std::vector<std::shared_ptr<nn_compiler::ir::NNLayer>> remove_layers_;
};

}  // namespace frontend
}  // namespace nn_compiler
