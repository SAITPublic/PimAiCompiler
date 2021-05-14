/*
 * Copyright (C) 2019 Samsung Electronics Co. LTD
 *
 * This software is proprietary of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#include "ir/ir_exec_step.hpp"
#include "ir/common/log.hpp"
#include "ir/ir_includes.hpp"
#include "ir/ir_tools.hpp"

namespace nn_compiler {

/**
 * @brief.      createInstructionFromIR
 * @details.    This function creates Instruction instance from IR
 * @param[in].  instr A instruction of flatbuffer
 * @param[out].
 * @returns.    return code
 */
std::unique_ptr<nn_ir::Instruction> IRExecStep::createInstruction(const IR::TargetHardware::Instruction* instr,
                                                                  const nn_ir::NNIR&                     graph) {
    std::unique_ptr<nn_ir::Instruction> g_instr;

    auto ir_id   = instr->id();
    auto ir_type = instr->instr_type();

    switch (ir_type) {
        case IR::TargetHardware::AnyInstr_ComputeInstr: {
            auto ir_compute_instr = instr->instr_as_ComputeInstr();
            auto ir_compute_type  = ir_compute_instr->compute_instr_type();

            switch (ir_compute_type) {
                case IR::TargetHardware::ComputeInstruction::AnyType_ExecuteStartInstr: {
                    g_instr =
                        std::make_unique<nn_ir::ExecuteStartInstruction>(ir_id, nn_ir::InstructionType::COMPUTE_START);
                    break;
                }
                case IR::TargetHardware::ComputeInstruction::AnyType_ExecuteSyncInstr: {
                    auto ir_sync_instr = ir_compute_instr->compute_instr_as_ExecuteSyncInstr();
                    auto start_id      = ir_sync_instr->start_id();
                    g_instr            = std::make_unique<nn_ir::ExecuteSyncInstruction>(
                        ir_id, nn_ir::InstructionType::COMPUTE_SYNC, start_id);
                    break;
                }
                default: {
                    Log::IR::E() << "IRExecStep::createInstruction() => unknown compute instruction type!";
                }
            }
            break;
        }
        case IR::TargetHardware::AnyInstr_MemoryInstr: {
            auto ir_memory_instr = instr->instr_as_MemoryInstr();
            auto ir_memory_type  = ir_memory_instr->memory_instr_type();

            switch (ir_memory_type) {
                case IR::TargetHardware::MemoryInstruction::AnyType_DMAStartInstr: {
                    auto ir_start_instr = ir_memory_instr->memory_instr_as_DMAStartInstr();

                    auto dma_ch_id    = ir_start_instr->dma_ch_id();
                    auto src_mem_info = ir_start_instr->src_mem();
                    auto dst_mem_info = ir_start_instr->dst_mem();

                    nn_ir::MemoryDataType       data_type    = nn_ir::MemoryDataType::PSUM;
                    nn_ir::IR_Node_Config_Type_ ir_data_type = src_mem_info->data_type();
                    data_type = std::get<nn_ir::MemoryDataType>(nn_ir::parseConfigType(ir_data_type));

                    uint32_t            sram_id = 0, size = 0;
                    auto                dir = ir_start_instr->direction();
                    nn_ir::DMADirection nn_dir;
                    if (dir == IR::TargetHardware::MemoryInstruction::DirectionType_DRAM2SRAM ||
                        dir == IR::TargetHardware::MemoryInstruction::DirectionType_DRAM2FIFO) {
                        nn_dir = dir == IR::TargetHardware::MemoryInstruction::DirectionType_DRAM2SRAM
                                     ? nn_ir::DMADirection::DRAM2SRAM
                                     : nn_ir::DMADirection::DRAM2FIFO;
                        sram_id = dst_mem_info->mem_id();
                        size    = dst_mem_info->size();
                    } else {
                        nn_dir  = nn_ir::DMADirection::SRAM2DRAM;
                        sram_id = src_mem_info->mem_id();
                        size    = src_mem_info->size();
                    }

                    auto src_addr = src_mem_info->addr();
                    auto dst_addr = dst_mem_info->addr();

                    g_instr = std::make_unique<nn_ir::DMAStartInstruction>(ir_id,
                                                                           nn_ir::InstructionType::DMA_START,
                                                                           nn_dir,
                                                                           data_type,
                                                                           dma_ch_id,
                                                                           sram_id,
                                                                           size,
                                                                           src_addr,
                                                                           dst_addr,
                                                                           parseDataLayout(src_mem_info->layout()),
                                                                           parseDataLayout(dst_mem_info->layout()));
                    break;
                }
                case IR::TargetHardware::MemoryInstruction::AnyType_DMASyncInstr: {
                    auto ir_sync_instr = ir_memory_instr->memory_instr_as_DMASyncInstr();
                    auto start_id      = ir_sync_instr->start_id();
                    g_instr =
                        std::make_unique<nn_ir::DMASyncInstruction>(ir_id, nn_ir::InstructionType::DMA_SYNC, start_id);
                    break;
                }
                default: {
                    Log::IR::E() << "IRExecStep::createInstruction() => unknown memory instruction type!";
                }
            }
            break;
        }
        case IR::TargetHardware::AnyInstr_MiscInstr: {
            auto ir_misc_instr = instr->instr_as_MiscInstr();
            auto ir_misc_type  = ir_misc_instr->misc_instr_type();

            switch (ir_misc_type) {
                case IR::TargetHardware::MiscInstruction::AnyType_SigSendInstr: {
                    const IR::TargetHardware::MiscInstruction::SigSendInstr* ir_send_instr =
                        ir_misc_instr->misc_instr_as_SigSendInstr();

                    g_instr = std::make_unique<nn_ir::SignalSendInstruction>(ir_id, ir_send_instr->dma_ch_id());
                    break;
                }
                case IR::TargetHardware::MiscInstruction::AnyType_SigWaitInstr: {
                    auto                    ir_wait_instr = ir_misc_instr->misc_instr_as_SigWaitInstr();
                    auto                    ir_send_ids   = ir_wait_instr->sender_id();
                    auto                    raw_data      = ir_send_ids->data();
                    auto                    data_size     = ir_send_ids->size();
                    std::vector<INSTR_ID_T> send_ids;
                    send_ids.assign(raw_data, raw_data + data_size);
                    g_instr = std::make_unique<nn_ir::SignalWaitInstruction>(
                        ir_id, nn_ir::InstructionType::SIG_WAIT, send_ids);
                    break;
                }
                default: {
                    Log::IR::E() << "IRExecStep::createInstruction() => unknown misc instruction type!";
                }
            }
            break;
        }
        case IR::TargetHardware::AnyInstr_VSyncInstr: {
            g_instr = std::make_unique<nn_ir::VsyncInstruction>(ir_id, nn_ir::InstructionType::VSYNC);
            break;
        }
        default:
            Log::IR::E() << "IRExecStep::createInstruction() => unknown instruction type!";
    }

    return g_instr;
}

/**
 * @brief.      createNodeExecutionStepFromIR
 * @details.    This function creates ExecutionStep instance from IR
 * @param[in].  step A step of flatbuffer
 * @param[out].
 * @returns.    return code
 */
std::unique_ptr<nn_ir::NodeExecutionStep>
IRExecStep::createNodeExecutionStep(const IR::TargetHardware::ExecutionStep* step, const nn_ir::NNIR& graph) {
    std::unique_ptr<nn_ir::NodeExecutionStep> g_step;

    auto                     ir_id   = step->id();
    auto                     ir_type = step->step_type();
    nn_ir::ExecutionStepType nn_type;

    if (ir_type == IR::TargetHardware::AnyExecutionStep_NodeExecutionStep) {
        nn_type = nn_ir::ExecutionStepType::NODE;
    } else if (ir_type == IR::TargetHardware::AnyExecutionStep_EdgeExecutionStep) {
        nn_type = nn_ir::ExecutionStepType::EDGE;
    } else {
        Log::IR::E() << "IRExecStep::createNodeExecutionStep() => unknown execution step type!";
    }

    Log::IR::E_IF(ir_type != IR::TargetHardware::AnyExecutionStep_NodeExecutionStep)
        << "IRExecStep::createNodeExecutionStep() => unknown execution step type!";
    auto                         node_step = step->step_as_NodeExecutionStep();
    auto                         node_id   = node_step->node_id();
    auto                         node_type = node_step->node_step();
    nn_ir::NodeExecutionStepType nn_node_type;
    switch (node_type) {
        case IR::TargetHardware::Type::NodeExecutionType_NODE_DATA_LOAD_START:
            nn_node_type = nn_ir::NodeExecutionStepType::NODE_DATA_LOAD_START;
            break;
        case IR::TargetHardware::Type::NodeExecutionType_NODE_DATA_LOAD_SYNC:
            nn_node_type = nn_ir::NodeExecutionStepType::NODE_DATA_LOAD_SYNC;
            break;
        case IR::TargetHardware::Type::NodeExecutionType_EXEC_START:
            nn_node_type = nn_ir::NodeExecutionStepType::EXEC_START;
            break;
        case IR::TargetHardware::Type::NodeExecutionType_EXEC_SYNC:
            nn_node_type = nn_ir::NodeExecutionStepType::EXEC_SYNC;
            break;
        default:
            Log::IR::E() << "IRExecStep::createNodeExecutionStep() => unknown node execution step type!";
    }
    g_step = std::make_unique<nn_ir::NodeExecutionStep>(graph, ir_id, nn_type, node_id, nn_node_type);
    addStepInstructions(step, g_step.get(), graph);
    return g_step;
}

/**
 * @brief.      createEdgeExecutionStepFromIR
 * @details.    This function creates ExecutionStep instance from IR
 * @param[in].  step A step of flatbuffer
 * @param[out].
 * @returns.    return code
 */
std::unique_ptr<nn_ir::EdgeExecutionStep>
IRExecStep::createEdgeExecutionStep(const IR::TargetHardware::ExecutionStep* step, const nn_ir::NNIR& graph) {
    std::unique_ptr<nn_ir::EdgeExecutionStep> g_step;

    auto                     ir_id   = step->id();
    auto                     ir_type = step->step_type();
    nn_ir::ExecutionStepType nn_type;

    if (ir_type == IR::TargetHardware::AnyExecutionStep_NodeExecutionStep) {
        nn_type = nn_ir::ExecutionStepType::NODE;
    } else if (ir_type == IR::TargetHardware::AnyExecutionStep_EdgeExecutionStep) {
        nn_type = nn_ir::ExecutionStepType::EDGE;
    } else {
        Log::IR::E() << "IRExecStep::createEdgeExecutionStep() => unknown execution step type!";
    }

    Log::IR::E_IF(ir_type != IR::TargetHardware::AnyExecutionStep_EdgeExecutionStep)
        << "IRExecStep::createEdgeExecutionStep() => unknown execution step type!";
    auto                         edge_step = step->step_as_EdgeExecutionStep();
    auto                         edge_id   = edge_step->edge_id();
    auto                         edge_type = edge_step->edge_step();
    nn_ir::EdgeExecutionStepType nn_edge_type;
    switch (edge_type) {
        case IR::TargetHardware::Type::EdgeExecutionType_LOAD_START:
            nn_edge_type = nn_ir::EdgeExecutionStepType::LOAD_START;
            break;
        case IR::TargetHardware::Type::EdgeExecutionType_LOAD_SYNC:
            nn_edge_type = nn_ir::EdgeExecutionStepType::LOAD_SYNC;
            break;
        case IR::TargetHardware::Type::EdgeExecutionType_STORE_START:
            nn_edge_type = nn_ir::EdgeExecutionStepType::STORE_START;
            break;
        case IR::TargetHardware::Type::EdgeExecutionType_STORE_SYNC:
            nn_edge_type = nn_ir::EdgeExecutionStepType::STORE_SYNC;
            break;
        default:
            Log::IR::E() << "IRExecStep::createEdgeExecutionStep() => unknown edge execution step type!";
    }
    g_step = std::make_unique<nn_ir::EdgeExecutionStep>(graph, ir_id, nn_type, edge_id, nn_edge_type);
    addStepInstructions(step, g_step.get(), graph);
    return g_step;
}
} // namespace nn_compiler
