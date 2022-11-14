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
/** @Details:
 *  There are multiple subgraphs in a NNModel after model builder phase of frontend importer.
 *  Many Ops are connected across different subgraphs, which makes the whole graph complicated,
 *  and difficult for optimization. So this pass is designed to reorgonize all subgraphs to one main graph.
 **/
class TakeInBodyNet : public Pass
{
   public:
    TakeInBodyNet();

    void fitIfCondition(std::unique_ptr<nn_compiler::ir::NNModel>& model);

    void fitLoopCondition(std::unique_ptr<nn_compiler::ir::NNModel>& model);

    bool fitCondition(std::unique_ptr<nn_compiler::ir::NNModel>& model);

    void run(std::unique_ptr<nn_compiler::ir::NNModel>& model);

    ~TakeInBodyNet() = default;

   private:
    std::vector<std::pair<std::shared_ptr<nn_compiler::ir::NNLayer>, std::shared_ptr<nn_compiler::ir::NNGraph>>>
        prim_if_layers_;

    std::vector<std::pair<std::shared_ptr<nn_compiler::ir::NNLayer>, std::shared_ptr<nn_compiler::ir::NNGraph>>>
        prim_loop_layers_;

    void take_in_if_body(std::unique_ptr<nn_compiler::ir::NNModel>& model);

    void take_in_loop_body(std::unique_ptr<nn_compiler::ir::NNModel>& model);

    uint32_t getUniqueTensorId(std::unique_ptr<nn_compiler::ir::NNModel>& model);
};

}  // namespace frontend
}  // namespace nn_compiler
