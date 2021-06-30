#include "compiler/include/middlend/passes/pass_builder.hpp"

// add pass files
#include "compiler/include/middlend/passes/graph/device_labeling.hpp"

#include <experimental/filesystem>
#include <unistd.h>

namespace nn_compiler {

template <typename... ArgTs>
void PassBuilder<ArgTs...>::getPassConfig(conf_json::Document& config,
                                          cl_opt::Option<std::string>& pass_config_file_path) const {
    std::string current_path = std::experimental::filesystem::current_path();
    if (current_path.find("/build") != std::string::npos) {
        std::string build_path = "../" + static_cast<std::string>(pass_config_file_path);
        pass_config_file_path.setValue(build_path);
    } else if (current_path.find("/build/compiler") != std::string::npos) {
        std::string build_path = "../../" + static_cast<std::string>(pass_config_file_path);
        pass_config_file_path.setValue(build_path);
    }

    std::ifstream config_file(pass_config_file_path);
    if (!config_file) {
        Log::ME::E() << LOG_PREFIX() "Failed to open pass configuration file! Please run at base or build directory.";
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
