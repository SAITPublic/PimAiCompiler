#include "examples/runtime/include/pipeline_manager.hpp"
#include "third_party/cmdline/cmdline.h"

int main(int argc, char* argv[])
{
    cmdline::parser command_line_parser;

    command_line_parser.add<std::string>("input", 'i', "Input file path", true, "");
    command_line_parser.add<std::string>("model", 'm', "Model type (RNNT/GNMT/HWR/Transfomer/SwitchTransformer)", true, "");
    command_line_parser.add<bool>("profiling", 'p', "Profiling", false, false);
    command_line_parser.add<int>("gpu_num", 'n', "Profiling", false, 1);

    command_line_parser.parse_check(argc, argv);

    // parse command arguments
    auto input_file_path = command_line_parser.get<std::string>("input");
    auto model_type = command_line_parser.get<std::string>("model");
    auto profiling = command_line_parser.get<bool>("profiling");
    auto gpu_num = command_line_parser.get<int>("gpu_num");

    examples::PipelineManager pipeline_manager;

    pipeline_manager.initialize(input_file_path, model_type, profiling,gpu_num);

    pipeline_manager.run();

    pipeline_manager.finalize();

    return 0;
}
