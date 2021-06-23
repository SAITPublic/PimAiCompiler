#include "compiler/include/middlend/context/compilation_context_builder.hpp"

// Add new CompilationContext file here

namespace nn_compiler {

void CtxBuilder::registerPassData(const std::string&  pass_name,
                                  const std::string&  data_name,
                                  CtxDataMode         mode,
                                  CtxDataScope        scope,
                                  CompilationContext& ctx) const {
#define PASS_DATA(DATA_TYPE)                               \
    if (data_name == #DATA_TYPE) {                         \
        ctx.createData<DATA_TYPE>(pass_name, mode, scope); \
        return;                                            \
    }
#include "compiler/include/middlend/context/PassData.def"

    Log::ME::E() << "Invalid pass data name `" << data_name.c_str() << '`';
}

} // namespace nn_compiler
