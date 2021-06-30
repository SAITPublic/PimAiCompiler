#include "common/include/command_line_parser.hpp"

#include "compiler/include/common/log.hpp"
#include "ir/include/ir_importer.hpp"

#include "compiler/include/middlend/middlend_driver.hpp"

/// Command-line options
static cl_opt::Option<int> verify_level("-v",
                                        "<level>",
                                        "verification level. Possible values: 0, 1, 2 (default: 0)",
                                        cl_opt::Required::NO,
                                        cl_opt::Hidden::NO,
                                        static_cast<int>(nn_compiler::PassManager<>::VerificationLevelType::NONE),
                                        std::vector<std::string>{"0", "1", "2"});

static cl_opt::Option<std::string>
        pass_config_file_path("-c",
                              "<configuration file>",
                              "pass config file path, default: compiler/include/middlend/passes/pass_config.json",
                              cl_opt::Required::NO,
                              cl_opt::Hidden::NO,
                              "compiler/include/middlend/passes/pass_config.json");

namespace nn_compiler {
namespace middlend {

RetVal MiddlendDriver::initialize(const std::string& in_ir_file_path) {
    in_ir_file_path_ = in_ir_file_path;
    importIR();

    return RetVal::SUCCESS;
}

RetVal MiddlendDriver::initialize(const std::vector<std::shared_ptr<nn_compiler::nn_ir::NNIR>>& NNIR_graphs) {
    for(auto iter = NNIR_graphs.begin(); iter != NNIR_graphs.end(); iter++) {
        graphs_.push_back(*iter);
    }

    return RetVal::SUCCESS;
}

RetVal MiddlendDriver::build() {
    Log::ME::I() << "NNCompiler MiddlendDriver::build() is called";
    buildPasses();

    return RetVal::SUCCESS;
}

RetVal MiddlendDriver::run() {
    Log::ME::I() << "NNCompiler MiddlendDriver::run() is called";

    // #1. Call the passes pipeline
    base_pass_manager_.initialize(base_util_manager_, base_trait_manager_);

    for (const auto& graph : graphs_) {
        base_pass_manager_.capability_check(*graph);
    }

    for (const auto& graph : graphs_) {
        base_context_.resetLocalData();

        base_pass_manager_.run(*graph, base_context_, verify_level);
    }

    base_pass_manager_.finalize();

    return RetVal::SUCCESS;
}

RetVal MiddlendDriver::wrapup() {
    Log::ME::I() << "NNCompiler MiddlendDriver::wrapup() is called";
    exportIR();

    return RetVal::SUCCESS;
}

RetVal MiddlendDriver::finalize() {
    return RetVal::SUCCESS;
}

void MiddlendDriver::importIR() {
    IRImporter ir_importer;
    ir_importer.getNNIRFromFile(in_ir_file_path_, graphs_);
}

void MiddlendDriver::buildPasses() {
    PassBuilder<> pass_builder;

    conf_json::Document config;
    pass_builder.getPassConfig(config, pass_config_file_path);

    if (!config.isObject()) {
        buildPasses(config, pass_builder);
    } else if (auto passes = config.get("passes", conf_json::Value())) {
        buildPasses(passes, pass_builder);
    } else {
        Log::ME::E() << LOG_PREFIX() "Invalid passes configuration file";
    }
}

template <typename... ArgTs>
void MiddlendDriver::buildPasses(const conf_json::Value&      root,
                                 const PassBuilder<ArgTs...>& pass_builder) {
    UtilBuilder util_builder;
    CtxBuilder  ctx_builder;

    for (const auto& node : root) {
        if (node.getBool("enable", false)) {
            auto type_name = node.getString("type", "");
            auto pass_name = node.getString("name", "");

            if (auto utils = node.get("utils", conf_json::Value())) {
                for (const auto& util_node : utils) {
                    auto util_name = util_node.asString();
                    util_builder.registerUtil(pass_name, util_name, base_util_manager_);
                }
            }

            if (auto data = node.get("ctx_data", conf_json::Value())) {
                for (const auto& data_node : data) {
                    auto data_name  = data_node.getString("name", "");
                    auto data_mode  = data_node.getString("mode", "");
                    auto data_scope = data_node.getString("scope", "");
                    auto mode       = strToCtxDataMode(data_mode);
                    auto scope      = strToCtxDataScope(data_scope);
                    ctx_builder.registerPassData(pass_name, data_name, mode, scope, base_context_);
                }
            }

            if (auto traits = node.get("traits", conf_json::Value())) {
                for (const auto& trait_node : traits) {
                    auto trait_name  = trait_node.getString("key", "");
                    auto trait_value = trait_node.getString("value", "");
                    base_trait_manager_.createTrait(pass_name, trait_name, trait_value);
                }
            }

            pass_builder.registerPass(pass_name, base_pass_manager_);

            bumpPassCounter();
        } else {
            Log::ME::E_IF(node.getBool("is_mandatory", false))
                    << "Mandatory passes cannot be disabled: " << node.getString("name", "");
        }
    }
}

void MiddlendDriver::exportIR() {
}

} // namespace middlend
} //namespace nn_compiler
