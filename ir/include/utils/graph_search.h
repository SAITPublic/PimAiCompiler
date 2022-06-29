#pragma once

#include "ir/include/nn_model.h"

namespace nn_compiler
{
namespace ir
{
namespace utils
{
std::vector<std::shared_ptr<ir::NNLayer>> searchPredecessor(const std::shared_ptr<ir::NNLayer> layer,
                                                            const std::unique_ptr<ir::NNModel> &nn_model);

std::vector<std::shared_ptr<ir::NNLayer>> searchSuccessorLayers(const std::shared_ptr<ir::NNLayer> layer,
                                                                const std::unique_ptr<ir::NNModel> &nn_model);

std::map<std::shared_ptr<ir::NNLayer>, uint32_t> searchMapSuccessor(const std::shared_ptr<ir::NNLayer> layer,
                                                                    const std::unique_ptr<ir::NNModel> &nn_model);

std::map<std::shared_ptr<ir::NNLayer>, std::vector<uint32_t>> searchMapSuccessors(
    const std::shared_ptr<ir::NNLayer> layer, const std::unique_ptr<ir::NNModel> &nn_model);

}  // namespace utils
}  // namespace ir
}  // namespace nn_compiler
