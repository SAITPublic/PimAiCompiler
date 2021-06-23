#include "compiler/include/middlend/passes/pass_service/conf_json/utility.hpp"
#include "compiler/include/middlend/common/log.hpp"

#include "common/include/algorithm_ext.hpp"
#include "common/include/utility.hpp"

namespace conf_json {

Option::Type typeFromStr(const std::string& str) {
#define OPT(ENUM, NAME, TYPE) \
    if (str == NAME) {        \
        return Option::ENUM;  \
    } else // NOLINT
#include "option_types.def"
    assert(false);

    return Option::null_opt;
}

std::string strFromType(Option::Type type) {
    switch (type) {
#define OPT(ENUM, NAME, TYPE) \
    case Option::ENUM:        \
        return NAME;
#include "option_types.def"
        default:
            assert(false);
    }

    return "";
}

template <typename ValueT>
Option::Value stringToValue(const std::string& str) {
    assert(false); // Unsupported conversion
    return Option::Value();
}

template <>
Option::Value stringToValue<bool>(const std::string& str) {
    static std::array<std::string, 3> false_values = {"false", "off", "0"};
    std::string                       lowered_str;

    std::transform(str.begin(), str.end(), std::back_inserter(lowered_str), ::tolower);
    return !estd::contains(false_values, lowered_str);
}

template <>
Option::Value stringToValue<std::string>(const std::string& str) {
    return str;
}

Option::Value stringToValue(Option::Type type, const std::string& str) {
    switch (type) {
#define OPT(ENUM, NAME, TYPE) \
    case Option::ENUM:        \
        return stringToValue<TYPE>(str);
#include "option_types.def"
        default:
            assert(false);
    }

    return Option::Value();
}

template <typename ValueT>
auto jsonFromValue(const Option::Value& value) {
    return std::get<ValueT>(value);
}

template <>
auto jsonFromValue<std::monostate>(const Option::Value& value) {
    return nn_compiler::Json::Value();
}

nn_compiler::Json::Value jsonFromValue(const Option::Value& value) {
    switch (static_cast<Option::Type>(value.index())) {
#define OPT(ENUM, NAME, TYPE) \
    case Option::ENUM:        \
        return jsonFromValue<TYPE>(value);
#include "option_types.def"
        default:
            assert(false);
    }

    return nn_compiler::Json::Value();
}

Option::Value valueFromJson(const nn_compiler::Json::Value& value) {
    using JsonValueType = nn_compiler::Json::ValueType;

    switch (value.type()) {
        case JsonValueType::booleanValue:
            return value.asBool();

        case JsonValueType::stringValue:
            return value.asString();

        default:
            return Option::Value();
    }
}

std::string availableOptionsInfo(const OptionsMap& options) {
    std::stringstream ss;
    const auto&       regular_names = options.names(OptionsMap::Filter::regular);

    if (!regular_names.empty()) {
        ss << "Available options are: " << estd::strJoin(regular_names, ", ");
    }

    const auto& debug_names = options.names(OptionsMap::Filter::debug);
    if (!debug_names.empty()) {
        if (!regular_names.empty()) {
            ss << "\n\t";
        }

        ss << "Available internal debug options (don't use for release) are: "
           << estd::strJoin(options.names(OptionsMap::Filter::debug), ", ");
    }

    return ss.str();
}

} // namespace conf_json
