#pragma once

#include "compiler/include/common/pass.hpp"
#include "compiler/include/frontend/optimizer/utils/attribute_helper.h"
#include "ir/include/nn_network.h"

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
                   const std::shared_ptr<nn_compiler::ir::NNNetwork> &graph,
                   std::shared_ptr<nn_compiler::ir::DTensor> &data, bool &remove_layer);

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
