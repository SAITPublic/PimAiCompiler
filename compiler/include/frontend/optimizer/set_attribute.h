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
#include "frontend/optimizer/utils/attribute_helper.h"

namespace nn_compiler
{
namespace frontend
{
/** @Details:
 *  Most constants are used for store attribute values of layers.
 *  If it is possible, set attribute value into layers and then remove constant layers (in the next
 *  remove_constant_layers pass), in order to simplify the graph and make following optimization more clear.
 **/
class SetAttribute : public Pass
{
   public:
    SetAttribute() { helper_ = std::make_shared<optimizer_utils::AttributeHelper>(); }

    bool fitCondition(std::unique_ptr<nn_compiler::ir::NNModel> &model);

    void run(std::unique_ptr<nn_compiler::ir::NNModel> &model);

    void doProcess(const std::shared_ptr<nn_compiler::ir::NNLayer> &layer,
                   std::unique_ptr<nn_compiler::ir::NNModel> &model, std::shared_ptr<nn_compiler::ir::DTensor> &data,
                   bool &remove_layer);

    void postProcess();

    ~SetAttribute() = default;

   private:
    std::vector<std::shared_ptr<nn_compiler::ir::NNLayer>> constant_layers_;

    std::vector<std::shared_ptr<nn_compiler::ir::NNLayer>> variable_layers_;

    std::shared_ptr<optimizer_utils::AttributeHelper> helper_ = nullptr;

    /**
     * @breif elements of edge_remove_helper is a mapping between:
     *  .first : layer
     *  .second: index of layer's input edges, these edges are prepared to be removed.
     **/
    std::map<std::shared_ptr<nn_compiler::ir::NNLayer>, std::vector<uint32_t>> edge_remove_helper_;
};

}  // namespace frontend
}  // namespace nn_compiler
