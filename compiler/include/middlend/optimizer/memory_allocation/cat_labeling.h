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
namespace middlend
{
/** @Details:
 *  Set relative attributes of aten::cat layers for custom optmization pattern:
 *
 *           |             |
 *      aten::lstm1    aten::bmm
 *             \         /
 *              aten::cat
 *                  |
 **/
class CatLabeling : public Pass
{
   public:
    CatLabeling();

    bool fitCondition(std::unique_ptr<nn_compiler::ir::NNModel>& model);

    void run(std::unique_ptr<nn_compiler::ir::NNModel>& model);

    ~CatLabeling() = default;

   private:
    std::vector<std::shared_ptr<nn_compiler::ir::NNLayer>> cat_labeling_layers_;

    std::vector<int64_t> target_cat_ids_bmm_;

    std::vector<int64_t> target_cat_ids_lstm_;

    void getOffspring(std::vector<int64_t>& res, std::shared_ptr<nn_compiler::ir::NNGraph> graph,
                      std::shared_ptr<nn_compiler::ir::NNLayer> layer, ir::LayerType targetLayerType, int level);
};  // class CatLabeling

}  // namespace middlend
}  // namespace nn_compiler
