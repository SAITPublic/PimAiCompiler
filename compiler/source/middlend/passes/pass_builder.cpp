#include "compiler/include/middlend/passes/pass_builder.hpp"

// add pass files
#include "compiler/include/middlend/passes/graph/device_labeling.hpp"
#include "compiler/include/middlend/passes/graph/lstm_labeling.hpp"
#include "compiler/include/middlend/passes/graph/cat_labeling.hpp"
#include "compiler/include/middlend/passes/memory/memory_labeling.hpp"
#include "compiler/include/middlend/passes/stream/control_node_execution.hpp"

namespace nn_compiler {

template <typename... ArgTs>
void PassBuilder<ArgTs...>::getPassConfig(conf_json::Document& config,
                                          cl_opt::Option<std::string>& pass_config_file_path) const {
    std::ifstream config_file(pass_config_file_path);
    if (!config_file) {
        Log::ME::E() << LOG_PREFIX() "Failed to open pass configuration file! "
                                     "Please speficy correct path of pass config file in env path.";
    }

    Log::ME::I() << "Read passes configuration from " << pass_config_file_path.get();
    config_file >> config;
}

template <typename... ArgTs>
void PassBuilder<ArgTs...>::registerPass(const std::string& pass_name, PassManager<ArgTs...>& pass_manager) const {
#define OPTIMIZATION_PASS(PASS_TYPE)   \
    if (pass_name == PASS_TYPE::getName()) { \
        pass_manager.addPass(PASS_TYPE());   \
        return;                              \
    }
#include "compiler/include/middlend/passes/Passes.def"

    Log::ME::E() << "Invalid pass name `" << pass_name << '`';
}

// explicit instantiation of class to generate its method
template class PassBuilder<>;

} // namespace nn_compiler
