#include <array>
#include <memory>
#include <sstream>

#include "common/include/algorithm_ext.hpp"
#include "common/include/utility.hpp"

#include "compiler/include/middlend/common/log.hpp"
#include "compiler/include/middlend/passes/pass_service/conf_json/conf_json.hpp"
#include "compiler/include/middlend/passes/pass_service/conf_json/expr_eval.hpp"
#include "compiler/include/middlend/passes/pass_service/conf_json/utility.hpp"

#ifndef NDEBUG
#define ErrLog Log::ME::E() << LOG_PREFIX()
#else
#define LOG_PREFIX(X) X
#define ErrLog        Log::ME::E()
#endif

namespace conf_json {

void Option::setValueFromString(const std::string& str) { value_ = stringToValue(type_, str); }

/**
 * @brief Return vector of available options names
 * @param[in] filter output filter: return all, regular or debug names
 */
std::vector<std::string> OptionsMap::names(Filter filter) const {
    std::vector<std::string> result;
    result.reserve(size());

    if (filter == Filter::all) {
        estd::transform(*this, std::back_inserter(result), [](const auto& pair) { return pair.first; });
        return result;
    }

    bool is_debug = filter == Filter::debug;
    estd::transform_if(
        *this,
        std::back_inserter(result),
        [](const auto& pair) { return pair.first; },
        [is_debug](const auto& pair) { return pair.second.isDebug() == is_debug; });

    return result;
}

Value Value::get(const char* key, const Value& default_value) const {
    JsonValue* member;

    if (null() || !(member = find(json_value_, key))) {
        return default_value;
    }

    Value result(member, options_);
    result.applyOptions();

    return result;
}

std::string Value::getString(const char* key, const std::string& default_value) const {
    Value member = get(key, Value());
    return member.empty() ? default_value : member.json_value_->asString();
}

bool Value::getBool(const char* key, bool default_value) const {
    Value member = get(key, Value());
    return member.empty() ? default_value : member.json_value_->asBool();
}

std::string Value::asString() const {
    if (null()) {
        return "";
    }

    applyOptions();

    return json_value_->asString();
}

/// @brief Get defined in JSON document option. Abort if option undefined.
/// @param name option name
const Option& Value::option(const std::string& name) const {
    auto iter = options_->find(name);

    if (iter == options_->end()) {
        ErrLog << "Undefined option '" << name << "'\n\t" << availableOptionsInfo(*options_);
    }

    return std::get<Option>(*iter);
}

Value::JsonValue* Value::find(const JsonValue* v, const char* key) {
    try {
        return const_cast<JsonValue*>(v->find(key, key + strlen(key)));
    }
    catch (std::exception&) {
        ErrLog << "Search for members requires object value or null value\n";
    }

    return nullptr;
}

/// @brief Substitute options by actual values in contents of stored value
void Value::applyOptions() const {
    static const std::string_view start_marker = "${";
    static const std::string_view end_marker   = "}";

    if (!json_value_->isString()) {
        return;
    }

    std::string_view expr(json_value_->asCString());
    size_t           start_pos = expr.find(start_marker);
    size_t           end_pos   = expr.rfind(end_marker);

    if (start_pos == std::string_view::npos) {
        return;
    }

    if (start_pos != 0) {
        ErrLog << "Unexpected symbols '" << std::string_view(expr.data(), start_pos) << "' before expression\n\t"
               << expr.data() << "\n";
    } else if (end_pos != expr.size() - 1) {
        ErrLog << "Expected '" << end_marker << "' after expression\n\t" << expr.data() << "\n";
    }

    expr.remove_prefix(start_marker.size());
    expr.remove_suffix(end_marker.size());

    *json_value_ = jsonFromValue(expr::Interpreter::eval(*options_, expr));

    return;
}

/// @brief Parse JSON document contained in given buffer
void Document::parse(const char* buf) {
    nn_compiler::Json::CharReaderBuilder           builder;
    std::string                                    errs;
    std::unique_ptr<nn_compiler::Json::CharReader> reader(builder.newCharReader());

    if (!reader->parse(buf, buf + std::strlen(buf), &root_, &errs)) {
        ErrLog << "Invalid Json: " << errs << "\n";
    }

    parseOptions();
}

/// @brief Read JSON document from stream
std::istream& operator>>(std::istream& sin, Document& doc) {
    try {
        sin >> doc.root_;
    }
    catch (std::exception& e) {
        ErrLog << "Invalid Json: " << e.what() << "\n";
    }

    if (sin.fail()) {
        ErrLog << "Reading failure\n";
    }

    doc.parseOptions();

    return sin;
}

/// @brief Write JSON document to stream
std::ostream& operator<<(std::ostream& sout, const Document& doc) {
    sout << doc.root_;

    if (sout.fail()) {
        ErrLog << "Writing failure\n";
    }

    return sout;
}

void Document::parseOptions() {
    if (!root_.isObject()) {
        return;
    }

    if (auto* options = find(&root_, "options")) {
        parseOptions(options, false);
    }

    if (auto* debug_options = find(&root_, "debug_options")) {
        parseOptions(debug_options, true);
    }
}

void Document::parseOptions(const JsonValue* options_array, bool is_debug) {
    for (auto& json_opt : *options_array) {
        JsonValue* name = find(&json_opt, "name");
        JsonValue* type = find(&json_opt, "type");

        if (name == nullptr) {
            ErrLog << "Option name must be specified";
        } else if (type == nullptr) {
            ErrLog << "Unspecified type for option '" << name->asString() << "'\n";
        }

        Option& opt = options_[name->asString()];

        if (opt.type_ != Option::null_opt) {
            ErrLog << "Multiple defeniton of option '" << name->asString() << "'\n";
        }

        opt.type_     = typeFromStr(type->asString());
        opt.is_debug_ = is_debug;

        if (opt.type_ == Option::null_opt) {
            ErrLog << "Invalid type of option '" << name->asString() << "'\n";
        }

        JsonValue* default_value = find(&json_opt, "default");

        if (default_value == nullptr) {
            ErrLog << "Unspecified default value for option '" << name->asString() << "'\n";
        }

        opt.value_ = valueFromJson(*default_value);

        if (opt.value_.index() != opt.type_) {
            ErrLog << "Type of default value for option '" << name->asString() << "' mismatches option type\n";
        }
    }
}

} // namespace conf_json
