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
 *  A prim::Variable is able to store a vetor of DTensors. However, only when DTensors are tensor type
 *  that the vector is needed. When DTensors are stored for only one value (Int/Float/Bool), it is possible
 *  to convert the vector of DTensors to a single DTensor which is stored at index zero of the vector.
 *  For example, [[1], [2], [3], [4]] -> [1, 2, 3, 4], where a pair of square brackets stands for a DTensor.
 **/
class RemakeDTensorOfPrimVariable : public Pass
{
   public:
    RemakeDTensorOfPrimVariable() { helper_ = std::make_shared<optimizer_utils::AttributeHelper>(); }

    bool fitCondition(std::unique_ptr<nn_compiler::ir::NNModel>& model);

    void run(std::unique_ptr<nn_compiler::ir::NNModel>& model);

    ~RemakeDTensorOfPrimVariable() = default;

   private:
    std::vector<std::shared_ptr<nn_compiler::ir::NNLayer>> variable_layers_;

    std::shared_ptr<optimizer_utils::AttributeHelper> helper_ = nullptr;

    /** @ Details: check the data of a prim::Variable is used to set attribute or input for other layers.
     *  @ Return:
     *       true: set attribute for other layers.
     *       false: used as input of other layers.
     **/
    bool checkVariableUsage(const std::shared_ptr<nn_compiler::ir::NNLayer>& layer,
                            const std::unique_ptr<ir::NNModel>& nn_model,
                            const std::shared_ptr<nn_compiler::ir::DTensor>& data);

    template <typename T>
    T getSingleValue(std::shared_ptr<nn_compiler::ir::DTensor>& d_tensor)
    {
        T ret_value;
        auto data = d_tensor->getData<T>();
        if ((*data).size() == 0) {
            DLOG(FATAL) << "processing data of prim::Variable gets NONE";
        } else {
            ret_value = (*data)[0];
        }
        return ret_value;
    }
};
}  // namespace frontend
}  // namespace nn_compiler
