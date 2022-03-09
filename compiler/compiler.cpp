/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is proprietary of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#include "common/include/command_line_parser.hpp"
#include "nn_compiler.hpp"

static cl_opt::Option<std::string> input_file_path(std::vector<std::string>{"-i", "--input"}, "<file>",
                                                   "Input file path", cl_opt::Required::YES);

static cl_opt::Option<std::string> model_name(std::vector<std::string>{"-m", "--model"}, "<name>",
                                              "Model name[RNNT, GNMT, HWR]", cl_opt::Required::YES);

int main(int argc, char* argv[])
{
    // parse arguments
    cl_opt::CommandLineParser::getInstance().parseCommandLine(argc, argv);

    // get a compiler instance (singleton)
    nn_compiler::compiler::NNCompiler compiler;

    std::unique_ptr<nn_compiler::ir::NNModel> model = std::make_unique<nn_compiler::ir::NNModel>();

    // initialize
    compiler.initialize(static_cast<std::string>(input_file_path), static_cast<std::string>(model_name));

    // compile
    compiler.compile(model);

    // finalize
    compiler.finalize();

    return 0;
}
