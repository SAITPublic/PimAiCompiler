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

static cl_opt::Option<std::string>
input_file_path(std::vector<std::string>{"-i", "--input"}, "<file>", "Input file path", cl_opt::Required::YES);

static cl_opt::Option<std::string>
model_name(std::vector<std::string>{"-m", "--model"}, "<name>", "Model name[RNNT, GNMT, HWR]", cl_opt::Required::YES);

static cl_opt::Option<int> compile_level("-l",
                                         "<compile level>",
                                         "compile level. Possible values: 0 (frontend->middlend->backend),\n\
                                          1 (middlend->backend), 2 (backend) (default: 0)",
                                         cl_opt::Required::YES,
                                         cl_opt::Hidden::NO);

int main(int argc, char* argv[]) {
    // parse arguments
    cl_opt::CommandLineParser::getInstance().parseCommandLine(argc, argv);

    // get a compiler instance (singleton)
    nn_compiler::compiler::NNCompiler compiler;

    // initialize
    compiler.initialize(static_cast<int>(compile_level), static_cast<std::string>(input_file_path),
                        static_cast<std::string>(model_name));

    // compile
    compiler.compile();

    // finalize
    compiler.finalize();

    return 0;
}
