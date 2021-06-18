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
#include "compiler/include/middlend/middlend_driver.hpp"

int main(int argc, char* argv[]) {
    // parse arguments
    cl_opt::CommandLineParser::getInstance().parseCommandLine(argc, argv);

    // get a driver instance (singleton)
    nn_compiler::middlend::MiddlendDriver driver;

    // initialize
    driver.initialize();

    // import
    driver.importIR();

    // run passes
    driver.runPasses();

    // export
    driver.exportIR();

    // finalize
    driver.finalize();

    return 0;
}
