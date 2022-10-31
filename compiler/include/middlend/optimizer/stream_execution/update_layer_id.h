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