/*
 * Copyright (C) 2019 Samsung Electronics Co. LTD
 *
 * This software is proprietary of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#include "ir/generated/ir_generated.h"
#include "ir/nn_ir.hpp"

namespace nn_compiler {

class IRExecStep {
 public:
    /**
     * @brief.      Constructor of IRExecStep.
     * @details.    This function constructs IRExecStep
     * @param[in].
     * @param[out].
     * @returns.
     */
    IRExecStep() = default;

    IRExecStep(const IRExecStep&) = delete;
    IRExecStep(IRExecStep&&)      = delete;
    IRExecStep& operator=(const IRExecStep&) = delete;
    IRExecStep& operator=(IRExecStep&&) = delete;

    /**
     * @brief.      createInstructionFromIR
     * @details.    This function creates Instruction instance from IR
     * @param[in].  instr A instr of flatbuffer
     * @param[out].
     * @returns.    return code
     */
    std::unique_ptr<nn_ir::Instruction> createInstruction(const IR::TargetHardware::Instruction* instr,
                                                          const nn_ir::NNIR&                     graph);
    /**
     * @brief.      createExecutionStepFromIR
     * @details.    This function creates ExecutionStep instance from IR
     * @param[in].  step A step of flatbuffer
     * @param[out].
     * @returns.    return code
     */
    std::unique_ptr<nn_ir::NodeExecutionStep> createNodeExecutionStep(const IR::TargetHardware::ExecutionStep* step,
                                                                      const nn_ir::NNIR&                       graph);
    std::unique_ptr<nn_ir::EdgeExecutionStep> createEdgeExecutionStep(const IR::TargetHardware::ExecutionStep* step,
                                                                      const nn_ir::NNIR&                       graph);

 private:
    void addStepInstructions(const IR::TargetHardware::ExecutionStep* step,
                             nn_ir::ExecutionStep*                    g_step,
                             const nn_ir::NNIR&                       graph) {
        if (auto ir_instrs = step->instrs()) {
            std::vector<std::unique_ptr<nn_ir::Instruction>> instrs;
            for (auto ir_instr : *ir_instrs) {
                auto nn_instr = createInstruction(ir_instr, graph);
                instrs.push_back(std::move(nn_instr));
            }
            g_step->setInstructions(std::move(instrs));
        }
    }
}; // class IRExecStep
} // namespace nn_compiler
