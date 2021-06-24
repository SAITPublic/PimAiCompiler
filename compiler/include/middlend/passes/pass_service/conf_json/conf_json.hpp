#pragma once

#include <memory>
#include <unordered_map>
#include <variant>

#include "common/include/iterators.hpp"
#include "third_party/json/json/json.h"

namespace conf_json {

/// @brief Represents defined in the configuration file option
class Option {
 public:
    using Value = std::variant<std::monostate, bool, std::string>;
    enum Type { null_opt = 0, bool_opt, string_opt };

    Option() = default;
    Option(Type t, Value v) : type_(t), value_(v) {}

    Type type() const { return type_; }

    bool isDebug() const { return is_debug_; }

    const Value& value() const { return value_; }
    Value&       value() { return value_; }
    void         setValueFromString(const std::string& str);

    template <typename T>
    constexpr const T& valueAs() const {
        return std::get<T>(value_);
    }

 private:
    friend class Document;

    Type  type_     = null_opt;
    bool  is_debug_ = false;
    Value value_;
};

class OptionsMap : public std::unordered_map<std::string, Option> {
 public:
    using Base = std::unordered_map<std::string, Option>;
    using Base::Base;

    /// @brief Flags to control names() method output
    enum class Filter { all, regular, debug };

    /**
     * @brief Return vector of available options names
     * @param[in] filter output filter: return all, regular or debug names
     */
    std::vector<std::string> names(Filter filter = Filter::all) const;
};

/// @brief Represents a JSON value
class Value {
 public:
    class ConstIterator;

    Value()          = default;
    virtual ~Value() = default;

    Value get(const char* key, const Value& default_value) const;

    std::string getString(const char* key, const std::string& default_value) const;
    bool        getBool(const char* key, bool default_value) const;

    std::string asString() const;

    bool empty() const { return null() || json_value_->empty(); }

    explicit operator bool() { return !empty(); }

    bool isObject() { return !null() && json_value_->isObject(); }

    inline ConstIterator begin() const;
    inline ConstIterator end() const;

    /// @brief Get defined in JSON document option. Abort if option undefined.
    /// @param name option name
    const Option& option(const std::string& name) const;
    Option&       option(const std::string& name) { return const_cast<Option&>(std::as_const(*this).option(name)); }

 protected:
    using JsonValue = nn_compiler::Json::Value;

    Value(JsonValue* jv, OptionsMap* om) : json_value_(jv), options_(om) {}

    static JsonValue* find(const JsonValue* v, const char* key);

    /// @brief Substitute options by actual values in contents of stored value
    void applyOptions() const;

 private:
    bool null() const { return json_value_ == nullptr || options_ == nullptr; }

    friend class ConstIterator;

    JsonValue*  json_value_ = nullptr;
    OptionsMap* options_    = nullptr;
};

class Value::ConstIterator : public IteratorAdaptor<Value::ConstIterator, Value::JsonValue::iterator, const Value> {
    using JsonValueIterator = Value::JsonValue::iterator;
    using Super             = IteratorAdaptor<Value::ConstIterator, Value::JsonValue::iterator, const Value>;

 public:
    ConstIterator() = default;

    void increment() { Super::increment(); }

    const Value& dereference() const {
        v_.json_value_ = const_cast<JsonValue*>(&*base());
        return v_;
    }

 private:
    ConstIterator(Value v, JsonValueIterator i) : Super(i), v_(v) {}

    friend class Value;

    mutable Value v_;
};

Value::ConstIterator Value::begin() const { return ConstIterator(*this, json_value_->begin()); }

Value::ConstIterator Value::end() const { return ConstIterator(*this, json_value_->end()); }

/// @brief Represents entire configuration file, stores options
class Document : public Value {
 public:
    Document() : Value(&root_, &options_) {}

    /// @brief Parse JSON document contained in given buffer
    void parse(const char* buf);

    /// @brief Read JSON document from stream
    friend std::istream& operator>>(std::istream& sin, Document& doc);

    /// @brief Write JSON document to stream
    friend std::ostream& operator<<(std::ostream& sout, const Document& doc);

 private:
    void parseOptions();
    void parseOptions(const JsonValue* options_array, bool is_debug);

    JsonValue  root_;
    OptionsMap options_;
};

} // namespace conf_json
