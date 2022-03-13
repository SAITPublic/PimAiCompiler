#pragma once

#include "compiler/include/common/pass.hpp"

#include "new_ir/include/nn_model.h"
#include "new_ir/include/nn_network.h"
#include "new_ir/include/types.h"

namespace nn_compiler
{

namespace middlend
{

/** @Brief Details: 
     1.This pass is the last pass at graph level, which updates layers' ID to sorted increasing order.
     2. After this pass, layer's ID equals to its position in the layer vector of NNNetwork (class memeber: layers_).
     3. So member function: getLayerByPosition() of NNNetwork becomes a safe & fast method, when passing layer's ID as the postion.
 **/

class UpdateLayerId : public Pass
{
   public:
    UpdateLayerId();

    bool fitCondition(std::unique_ptr<nn_compiler::ir::NNModel>& model);

    void run(std::unique_ptr<nn_compiler::ir::NNModel>& model);

    ~UpdateLayerId() = default;

};  // class UpdateLayerId

}  // namespace middlend
}  // namespace nn_compiler
