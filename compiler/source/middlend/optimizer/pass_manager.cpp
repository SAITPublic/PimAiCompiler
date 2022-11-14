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

#include "middlend/optimizer/pass_manager.h"

#include "middlend/optimizer/memory_allocation/cat_labeling.h"
#include "middlend/optimizer/memory_allocation/lstm_labeling.h"
#include "middlend/optimizer/stream_execution/control_layer_execution.h"
#include "middlend/optimizer/stream_execution/multi_stream_execution.h"
#include "middlend/optimizer/stream_execution/update_layer_id.h"

namespace nn_compiler
{
namespace middlend
{
PassManager::PassManager() {}

void PassManager::runPasses(std::unique_ptr<nn_compiler::ir::NNModel>& model)
{
    DLOG(INFO) << "middlend PassManager::runPasses is called.";
    auto base_pass = std::make_shared<Pass>();

    auto multi_stream_execution = std::make_shared<MultiStreamExecution>();
    auto update_layer_id = std::make_shared<UpdateLayerId>();
    auto control_layer_execution = std::make_shared<ControlLayerExecution>();
    auto lstm_labeling = std::make_shared<LstmLabeling>();
    auto cat_labeling = std::make_shared<CatLabeling>();

    base_pass->add(multi_stream_execution);
    multi_stream_execution->add(update_layer_id);
    update_layer_id->add(control_layer_execution);
    control_layer_execution->add(lstm_labeling);
    lstm_labeling->add(cat_labeling);

    while (base_pass->getSuccessor() != nullptr) {
        base_pass = base_pass->getSuccessor();
        if (base_pass->fitCondition(model)) {
            base_pass->run(model);
        }
    }
}

}  // namespace middlend
}  // namespace nn_compiler
