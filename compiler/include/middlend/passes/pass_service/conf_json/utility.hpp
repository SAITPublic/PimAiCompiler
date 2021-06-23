#pragma once

#include <cassert>

#include "compiler/include/middlend/passes/pass_service/conf_json/conf_json.hpp"

namespace conf_json {

Option::Type typeFromStr(const std::string& str);
std::string  strFromType(Option::Type type);

Option::Value stringToValue(Option::Type type, const std::string& str);

Option::Value            valueFromJson(const nn_compiler::Json::Value& value);
nn_compiler::Json::Value jsonFromValue(const Option::Value& value);

std::string availableOptionsInfo(const OptionsMap& options);

} // namespace conf_json
