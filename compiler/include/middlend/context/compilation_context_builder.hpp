#pragma once

#include "compiler/include/middlend/context/compilation_context.hpp"

namespace nn_compiler {

/// @brief class for building compilation context
class CtxBuilder {
 public:
    void registerPassData(const std::string&  pass_name,
                          const std::string&  data_name,
                          CtxDataMode         mode,
                          CtxDataScope        scope,
                          CompilationContext& util_manager) const;
};
} // namespace nn_compiler
