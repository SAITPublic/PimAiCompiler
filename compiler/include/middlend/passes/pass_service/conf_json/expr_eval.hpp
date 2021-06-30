#pragma once

#include <optional>
#include <regex> // NOLINT

#include "compiler/include/common/log.hpp"
#include "compiler/include/middlend/passes/pass_service/conf_json/conf_json.hpp"

namespace conf_json {
namespace expr {

/// @brief Represents token of expressions with pass config options
struct Token {
    enum Type { option, op_not, op_and, op_or, skip, mismatch, expr_end };

    Type             type;
    std::string_view str; // token substring of the expression string being evaluated
};

/// @brief Lexical analyzer of expressions with pass config options
class Lexer {
 public:
    explicit Lexer(std::string_view expr) : expr_(expr), pos_(expr.begin()) {}

    std::string_view expr() { return expr_; }

    /// @brief Return next token of given expression
    Token next();

 private:
    std::string_view expr_;
    const char*      pos_;

    static std::vector<std::pair<Token::Type, std::regex>> token_regexprs_;
};

/// @brief Represents arguments stack for calclualting RPN expression
using Args = std::vector<Option::Value>;

/// @brief Base class for representing operator in expressions with pass config options
class BaseOperator {
 public:
    explicit BaseOperator(Token token) : token_(token) {}
    virtual ~BaseOperator() = default;

    const Token& token() const { return token_; }

    virtual size_t priority() const = 0;

    virtual size_t       argc() const    = 0;
    virtual Option::Type argType() const = 0;

    /**
     * @brief   Apply operator to given args.
     * @details Pop arguments from back of args and push result.
     * @returns std::optional that holds error message in case of failure.
     */
    virtual std::optional<std::string> apply(Args& args) const = 0;

 private:
    Token token_;
};

/// @brief Interpreter of expressions with pass config options
class Interpreter {
 public:
    /**
     * @brief  Evaluate expression with pass config options
     * @param  options pass config options
     * @param  expr expression to evaluate
     * @return Option holding result of the evaluation
     */
    static Option::Value eval(const OptionsMap& options, std::string_view expr);

 private:
    Interpreter(const OptionsMap& options, std::string_view expr) : options_(options), lexer_(expr) {}

    Option::Value eval();
    Option::Value getOptionValue(Token t);

    void handleOperator(std::unique_ptr<BaseOperator> op);
    void applyOperator(BaseOperator* op);

    void reportError(const std::string& what, Token t, const std::string info = "");

    const OptionsMap& options_;

    Lexer lexer_;

    Args args_;

    /// @brief Represents operators stack for Shunting-yard algorithm
    std::vector<std::unique_ptr<BaseOperator>> operators_;
};

class OperatorNOT : public BaseOperator {
 public:
    explicit OperatorNOT(Token token) : BaseOperator(token) {}

    size_t priority() const override { return 2; }

    size_t       argc() const override { return 1; }
    Option::Type argType() const override { return Option::bool_opt; };

    std::optional<std::string> apply(Args& args) const override;
};

class OperatorAND : public BaseOperator {
 public:
    explicit OperatorAND(Token token) : BaseOperator(token) {}

    size_t priority() const override { return 1; }

    size_t       argc() const override { return 2; }
    Option::Type argType() const override { return Option::bool_opt; };

    std::optional<std::string> apply(Args& args) const override;
};

class OperatorOR : public BaseOperator {
 public:
    explicit OperatorOR(Token token) : BaseOperator(token) {}

    size_t priority() const override { return 0; }

    size_t       argc() const override { return 2; }
    Option::Type argType() const override { return Option::bool_opt; };

    std::optional<std::string> apply(Args& args) const override;
};

} // namespace expr
} // namespace conf_json
