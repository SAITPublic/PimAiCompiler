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
namespace middlend
{
/** @Details:
 *  Set relative attributes of aten::LSTM1 layers for custom optmization pattern:
 *
 *                 |
 *            aten::lstm1
 *                 |
 *         prim::tuple_construct
 *                 |
 *            aten::append
 *                 |
 **/
class LstmLabeling : public Pass
{
   public:
    LstmLabeling();

    bool fitCondition(std::unique_ptr<nn_compiler::ir::NNModel>& model);

    void run(std::unique_ptr<nn_compiler::ir::NNModel>& model);

    ~LstmLabeling() = default;

   private:
    std::vector<std::shared_ptr<nn_compiler::ir::NNLayer>> aten_lstm1_layers_;
};  // class LstmLabeling

}  // namespace middlend
}  // namespace nn_compiler
