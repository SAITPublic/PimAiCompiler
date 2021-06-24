#pragma once

#include "compiler/include/middlend/passes/pass_manager.hpp"
#include "compiler/include/middlend/passes/pass_service/conf_json/conf_json.hpp"

namespace nn_compiler {

template <typename... ArgTs>
class PassBuilder {
 public:
    /**
     * @brief.      get PassConfig
     * @details.    This function parse and return Pass configuration
     */
    void getPassConfig(conf_json::Document& config, cl_opt::Option<std::string>& pass_config_file_path) const;

    /**
     * @brief.      register pass
     * @details.    This function registers Pass in PassManager
     */
    void registerPass(const std::string& pass_name, PassManager<ArgTs...>& pass_manager) const;

}; // class PassBuilder

extern template class PassBuilder<>;

} // namespace nn_compiler
