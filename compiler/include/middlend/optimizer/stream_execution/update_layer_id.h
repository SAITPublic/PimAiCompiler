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
/** @Brief Details:
 *   1.This pass is the last pass at graph level, which updates layers' ID to sorted increasing order.
 *   2.After this pass, layer's ID equals to its position in the layer vector of NNGraph (class memeber: layers_).
 *   3.So member function: getLayerByPosition() of NNGraph becomes a safe & fast method, when passing layer's ID as
 *     the postion.
 **/
class UpdateLayerId : public Pass
{
   public:
    UpdateLayerId();

    bool fitCondition(std::unique_ptr<nn_compiler::ir::NNModel>& model);

    void run(std::unique_ptr<nn_compiler::ir::NNModel>& model);

    void setLayerRelations(std::shared_ptr<ir::NNLayer>& layer, std::unique_ptr<nn_compiler::ir::NNModel>& model);

    ~UpdateLayerId() = default;

};  // class UpdateLayerId

}  // namespace middlend
}  // namespace nn_compiler
