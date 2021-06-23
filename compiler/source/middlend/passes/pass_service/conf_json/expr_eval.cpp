#include <iomanip>
#include <sstream>

#include "compiler/include/middlend/passes/pass_service/conf_json/expr_eval.hpp"
#include "compiler/include/middlend/passes/pass_service/conf_json/utility.hpp"

#include "common/include/algorithm_ext.hpp"
#include "common/include/utility.hpp"

#ifndef NDEBUG
#define ErrLog Log::ME::E() << LOG_PREFIX()
#else
#define LOG_PREFIX(X) X
#define ErrLog        Log::ME::E()
#endif

namespace conf_json {

extern std::string availableOptionsInfo(const OptionsMap& options);
namespace expr {

std::vector<std::pair<Token::Type, std::regex>> Lexer::token_regexprs_ = {
    {Token::option, std::regex("^\\w+", std::regex::ECMAScript | std::regex::optimize)},
    {Token::op_not, std::regex("^!", std::regex::ECMAScript | std::regex::optimize)},
    {Token::op_and, std::regex("^&&", std::regex::ECMAScript | std::regex::optimize)},
    {Token::op_or, std::regex("^\\|\\|", std::regex::ECMAScript | std::regex::optimize)},
    {Token::skip, std::regex("^\\s+", std::regex::ECMAScript | std::regex::optimize)},
    {Token::mismatch, std::regex("^.", std::regex::ECMAScript | std::regex::optimize)}};

/// @brief Return next token of given expression
Token Lexer::next() {
    if (pos_ == expr_.end()) {
        return Token{.type = Token::expr_end};
    }

    std::cmatch cm;
    Token::Type type = estd::find_if(token_regexprs_, [&](const auto& tr) {
                           return std::regex_search(pos_, expr_.end(), cm, tr.second);
                       })->first;

    pos_ = cm[0].second;

    if (type == Token::skip) {
        return next();
    } else if (type == Token::mismatch) {
        ErrLog << "Unexpected symbol '" << *(cm[0].first) << "' in expression\n\t" << expr_ << "\n\t"
               << std::setw(cm[0].second - expr_.data() + 1) << "^\n";
    }

    return Token{.type = type, .str = std::string_view(cm[0].first, cm.length())};
}

/**
 * @brief  Evaluate expression with pass config options
 * @param  options pass config options
 * @param  expr expression to evaluate
 * @return Option holding result of the evaluation
 */
Option::Value Interpreter::eval(const OptionsMap& options, std::string_view expr) {
    expr::Interpreter interpreter(options, expr);
    return interpreter.eval();
}

Option::Value Interpreter::eval() {
    // Transform expression to RPN using Shunting-yard algorithm and calculate result on the fly.

    Token::Type prev_token_type = Token::expr_end;
    for (auto token = lexer_.next(); token.type != Token::expr_end; token = lexer_.next()) {
        switch (token.type) {
            case Token::option: {
                if (prev_token_type == Token::option) {
                    reportError("Expected operator before", token);
                }

                args_.emplace_back(getOptionValue(token));
                break;
            }
            case Token::op_not: {
                handleOperator(std::make_unique<OperatorNOT>(token));
                break;
            }
            case Token::op_and: {
                handleOperator(std::make_unique<OperatorAND>(token));
                break;
            }
            case Token::op_or: {
                handleOperator(std::make_unique<OperatorOR>(token));
                break;
            }
            default: {
                reportError("Unexpected token", token);
            }
        }

        prev_token_type = token.type;
    }

    for (auto i = operators_.rbegin(); i != operators_.rend(); ++i) {
        applyOperator(i->get());
    }

    return args_.front();
}

Option::Value Interpreter::getOptionValue(Token t) {
    auto iter = options_.find(std::string(t.str.data(), t.str.size()));

    if (iter == options_.end()) {
        reportError("Undefined option", t, availableOptionsInfo(options_));
    }

    return iter->second.value();
}

void Interpreter::handleOperator(std::unique_ptr<BaseOperator> op) {
    while (!operators_.empty() && operators_.back()->priority() >= op->priority()) {
        applyOperator(operators_.back().get());
        operators_.pop_back();
    }

    operators_.emplace_back(std::move(op));
}

void Interpreter::applyOperator(BaseOperator* op) {
    if (args_.size() < op->argc()) {
        std::stringstream info;
        info << "Expected: " << op->argc() << ", got: " << args_.size();

        reportError("Missing arguments for operator", op->token(), info.str());
    }

    auto op_arg = args_.begin() + (args_.size() - op->argc());
    for (size_t i = 0; i < op->argc(); i++, op_arg++) {
        if (op_arg->index() == op->argType()) {
            continue;
        }

        std::stringstream what;
        what << "Invalid type of argument number " << i + 1 << " for operator";

        std::stringstream info;
        info << "Expected: " << strFromType(op->argType())
             << ", got: " << strFromType(static_cast<Option::Type>(op_arg->index()));

        reportError(what.str(), op->token(), info.str());
    }

    op->apply(args_);
}

void Interpreter::reportError(const std::string& what, Token t, const std::string info) {
    std::string_view  expr = lexer_.expr();
    std::stringstream ss;

    ss << what << " '" << t.str << "' in expression:\n\t" << expr << "\n\t" << std::setw(t.str.data() - expr.data() + 2)
       << "^\n";

    if (!info.empty()) {
        ss << "\t" << info << "\n";
    }

    ErrLog << ss.str();
}

std::optional<std::string> OperatorNOT::apply(Args& args) const {
    args.back() = !std::get<bool>(args.back());
    return std::nullopt;
}

std::optional<std::string> OperatorAND::apply(Args& args) const {
    bool arg = std::get<bool>(args.back());
    args.pop_back();
    args.back() = std::get<bool>(args.back()) && arg;

    return std::nullopt;
}

std::optional<std::string> OperatorOR::apply(Args& args) const {
    bool arg = std::get<bool>(args.back());
    args.pop_back();
    args.back() = std::get<bool>(args.back()) || arg;

    return std::nullopt;
}

} // namespace expr
} // namespace conf_json
