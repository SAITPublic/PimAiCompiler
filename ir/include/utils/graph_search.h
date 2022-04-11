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

std::vector<std::shared_ptr<ir::NNLayer>> searchPredecessor(const std::shared_ptr<ir::NNLayer> layer,
                                                            const std::shared_ptr<ir::NNGraph> graph);

std::vector<std::shared_ptr<ir::NNLayer>> searchSuccessorLayerOnly(const std::shared_ptr<ir::NNLayer> layer,
                                                                   const std::shared_ptr<ir::NNGraph> graph);

std::map<std::shared_ptr<ir::NNLayer>, uint32_t> searchSuccessor(const std::shared_ptr<ir::NNLayer> layer,
                                                                 const std::shared_ptr<ir::NNGraph> graph);

std::map<std::shared_ptr<ir::NNLayer>, std::vector<uint32_t>> searchSuccessors(
    const std::shared_ptr<ir::NNLayer> layer, const std::shared_ptr<ir::NNGraph> graph);

std::shared_ptr<ir::NNLayer> searchLayerByOutID(uint32_t out_id, const std::shared_ptr<ir::NNGraph> graph);

}  // namespace utils
}  // namespace ir
}  // namespace nn_compiler
