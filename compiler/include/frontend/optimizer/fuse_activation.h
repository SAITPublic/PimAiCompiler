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
 *  Fuse activation layers to its predecessor addmm layer, as the computation of activation could
 *  be done in addmm layer directly.
 *
 *         |
 *   aten::addmm (no activation)               |
 *         |                               aten::addmm (with relu)
 *   aten::transpose           ---->           |
 *         |                             aten::transpose
 *     aten::relu                              |
 *         |
 **/
class FuseActivation : public Pass
{
   public:
    FuseActivation();

    bool fitCondition(std::unique_ptr<nn_compiler::ir::NNModel>& model);

    void run(std::unique_ptr<nn_compiler::ir::NNModel>& model);

    ~FuseActivation() = default;

   private:
    std::vector<std::string> supportd_host_types_ = {"aten::transpose"};
    bool feasibleHostType(const std::string& type);

    std::vector<std::string> supported_parasite_types_ = {"aten::relu", "aten::max"};

    std::vector<std::shared_ptr<nn_compiler::ir::NNLayer>> layers_;

    bool feasibleParasiteType(const std::string& type);
};

}  // namespace frontend
}  // namespace nn_compiler
