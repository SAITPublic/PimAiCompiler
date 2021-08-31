#include "common/include/command_line_parser.hpp"
#include "examples/runtime/include/pipeline_manager.hpp"

static cl_opt::Option<std::string>
input_file_path_option(std::vector<std::string>{"-i", "--input"}, "<file>",
                                                "Input file path",
                                                cl_opt::Required::YES);

static cl_opt::Option<std::string>
model_type_option(std::vector<std::string>{"-m", "--model"}, "<model type>",
                                           "Possible model type: RNNT/HWR/GNMT",
                                           cl_opt::Required::YES);

static cl_opt::Option<int>
compile_level_option("-l", "<compile level>",
                     "compile level. Possible values: 0 (frontend->middlend->backend),\n\
                     1 (middlend->backend), 2 (backend)",
                     cl_opt::Required::YES);

int main(int argc, char* argv[]) {
    cl_opt::CommandLineParser::getInstance().parseCommandLine(argc, argv);
    // parse command arguments
    auto input_file_path = static_cast<std::string>(input_file_path_option);
    auto model_type      = static_cast<std::string>(model_type_option);
    auto compile_level   = static_cast<int>(compile_level_option);

    examples::PipelineManager pipeline_manager;

    pipeline_manager.initialize(input_file_path, compile_level, model_type);

    pipeline_manager.run();

    pipeline_manager.finalize();

    return 0;
}
