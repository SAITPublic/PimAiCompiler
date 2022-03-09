#include "common/include/command_line_parser.hpp"
#include "examples/runtime/include/pipeline_manager.hpp"

static cl_opt::Option<std::string> input_file_path_option(std::vector<std::string>{"-i", "--input"}, "<file>",
                                                          "Input file path", cl_opt::Required::YES);

static cl_opt::Option<std::string> model_type_option(std::vector<std::string>{"-m", "--model"}, "<model type>",
                                                     "Possible model type: RNNT/HWR/GNMT", cl_opt::Required::YES);

static cl_opt::Option<bool> profiling_option("-p", "--profiling", "run with profiling", cl_opt::Required::NO,
                                             cl_opt::Hidden::NO, false);

int main(int argc, char* argv[])
{
    cl_opt::CommandLineParser::getInstance().parseCommandLine(argc, argv);
    // parse command arguments
    auto input_file_path = static_cast<std::string>(input_file_path_option);
    auto model_type = static_cast<std::string>(model_type_option);
    auto profiling = static_cast<bool>(profiling_option);

    examples::PipelineManager pipeline_manager;

    pipeline_manager.initialize(input_file_path, model_type, profiling);

    pipeline_manager.run();

    pipeline_manager.finalize();

    return 0;
}
