#include "compiler/include/middlend/utils/util_builder.hpp"

#include "compiler/include/middlend/utils/op_basic_util.hpp"

#include "common/include/utility.hpp"

namespace nn_compiler {

void UtilBuilder::registerUtil(const std::string& pass_name,
                               const std::string& util_name,
                               UtilManager&       util_manager) const {
    bool found = false;

    // When/if we add stateful utils
    // this check needs to be conditionalized on util's type
#define UTIL(UTIL_TYPE)                                                                   \
    if (util_name == #UTIL_TYPE) {                                                        \
        static_assert(estd::is_empty<UTIL_TYPE>, "Stateful utils are not yet supported"); \
        util_manager.addUtil(UTIL_TYPE(), pass_name);                                     \
        found = true;                                                                     \
    }
#include "compiler/include/middlend/utils/Utils.def"

#define BASE_UTIL(UTIL_TYPE, BASE_UTIL_TYPE)                                              \
    if (util_name == #UTIL_TYPE) {                                                        \
        static_assert(estd::is_empty<UTIL_TYPE>, "Stateful utils are not yet supported"); \
        util_manager.addUtilAsBase<BASE_UTIL_TYPE>(UTIL_TYPE(), pass_name);               \
        found = true;                                                                     \
    }
#include "compiler/include/middlend/utils/Utils.def"

    LOGE_IF(ME, !found, "Invalid util name `%s`\n", util_name.c_str());
}

} // namespace nn_compiler
