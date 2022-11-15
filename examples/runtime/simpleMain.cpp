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

#include "examples/runtime/include/pipeline_manager.hpp"
#include "third_party/cmdline/cmdline.h"

int main(int argc, char* argv[])
{
    cmdline::parser command_line_parser;

    command_line_parser.add<std::string>("input", 'i', "Input file path", true, "");
    command_line_parser.add<std::string>("model", 'm', "Model type (RNNT/GNMT/HWR/Transfomer/SwitchTransformer)", true,
                                         "");
    command_line_parser.add<bool>("profiling", 'p', "Profiling", false, false);
    command_line_parser.add<int>("gpu_num", 'n', "Profiling", false, 1);

    command_line_parser.parse_check(argc, argv);

    // parse command arguments
    auto input_file_path = command_line_parser.get<std::string>("input");
    auto model_type = command_line_parser.get<std::string>("model");
    auto profiling = command_line_parser.get<bool>("profiling");
    auto gpu_num = command_line_parser.get<int>("gpu_num");

    examples::PipelineManager pipeline_manager;

    pipeline_manager.initialize(input_file_path, model_type, gpu_num, profiling);

    pipeline_manager.run();

    pipeline_manager.finalize();

    return 0;
}
