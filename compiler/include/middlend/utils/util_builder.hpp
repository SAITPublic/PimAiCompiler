#pragma once

#include "compiler/include/middlend/utils/util_manager.hpp"

namespace nn_compiler {

/// @brief class for building utils
class UtilBuilder {
 public:
    void registerUtil(const std::string& pass_name, const std::string& util_name, UtilManager& util_manager) const;
};

} // namespace nn_compiler
