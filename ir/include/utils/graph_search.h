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
