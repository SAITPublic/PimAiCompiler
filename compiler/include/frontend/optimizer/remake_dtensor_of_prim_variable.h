#pragma once

#include "compiler/include/common/pass.hpp"
#include "compiler/include/frontend/optimizer/utils/attribute_helper.h"

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
    RemakeDTensorOfPrimVariable() { helper_ = std::make_shared<AttributeHelper>(); }

    bool fitCondition(std::unique_ptr<nn_compiler::ir::NNModel>& model);

    void run(std::unique_ptr<nn_compiler::ir::NNModel>& model);

    ~RemakeDTensorOfPrimVariable() = default;

   private:
    std::vector<std::shared_ptr<nn_compiler::ir::NNLayer>> variable_layers_;

    std::shared_ptr<AttributeHelper> helper_ = nullptr;

    /** @ Details: check the data of a prim::Variable is used to set attribute or input for other layers.
     *  @ Return:
     *       true: set attribute for other layers.
     *       false: used as input of other layers.
     **/
    bool checkVariableUsage(const std::shared_ptr<nn_compiler::ir::NNLayer>& layer,
                            const std::shared_ptr<nn_compiler::ir::NNNetwork>& graph,
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
