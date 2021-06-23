#pragma once

#include "common/include/common.hpp"
#include "common/include/types.hpp"
#include "ir/include/nn_ir.hpp"
#include "compiler/include/middlend/context/compilation_context.hpp"
#include "compiler/include/middlend/trait/trait_manager.hpp"
#include "compiler/include/middlend/utils/util_manager.hpp"

namespace nn_compiler {

/// @brief internal classes to support implementation of concept based polymorphism
namespace internal {

/// @brief interface for passes
template <typename... ArgTs>
class PassConcept {
 public:
    virtual ~PassConcept() = default;

    virtual void   capability_check(const nn_ir::NNIR& graph, ArgTs... args)                                     = 0;
    virtual RetVal initialize(const UtilManager& util_manager, const TraitManager& trait_manager, ArgTs... args) = 0;
    virtual RetVal run(nn_ir::NNIR& graph, CompilationContext& ctx, ArgTs... args)                               = 0;
    virtual void   finalize(ArgTs... args)                                                                       = 0;
    virtual RetVal postCheck(const nn_ir::NNIR& graph, ArgTs... args)                                            = 0;
    virtual RetVal preCheck(const nn_ir::NNIR& graph, ArgTs... args) const                                       = 0;
};

/// @brief This class is container for passes. All passes can be run if they implement
///        `initialize`, `run` and `finalize` methods.
///        * `run` method must take `NNIR` as first argument and `UtilManager` as second and return `RetVal` value
template <typename PassT, typename... ArgTs>
class PassModel final : public PassConcept<ArgTs...> {
 public:
    explicit PassModel(PassT pass) : pass_(std::move(pass)) {}

    void capability_check(const nn_ir::NNIR& graph, ArgTs... args) override {
        PassT::trace("capability_check(graph)");
        pass_.capability_check(graph, args...);
    }

    RetVal initialize(const UtilManager& util_manager, const TraitManager& trait_manager, ArgTs... args) override {
        PassT::trace("initialize(util_manager and trait_manager)");
        RetVal res = pass_.initialize(util_manager, trait_manager, args...);
        return res;
    }

    RetVal run(nn_ir::NNIR& graph, CompilationContext& ctx, ArgTs... args) override {
        std::string text = std::string("run(graph #") + std::to_string(graph.getId()) + ")";
        PassT::trace(text.c_str());
        RetVal res = pass_.run(graph, ctx, args...);
        return res;
    }

    void finalize(ArgTs... args) override { pass_.finalize(args...); }

    RetVal postCheck(const nn_ir::NNIR& graph, ArgTs... args) override {
        RetVal res = pass_.postCheck(graph, args...);
        return res;
    }

    RetVal preCheck(const nn_ir::NNIR& graph, ArgTs... args) const override {
        RetVal res = pass_.preCheck(graph, args...);
        return res;
    }

 private:
    PassT pass_;
};

} // namespace internal

/// @brief This is pass manager class that controls compilation pipeline. It allows to register passes
///        and runs all registered passes. This idiom is also implemented in new LLVM PassManager.
///        Implementation is based on concepts-based polymorphism that was introduced by Sean Parent.
///        See description of this idiom:
///        https://sean-parent.stlab.cc/papers-and-presentations/#value-semantics-and-concept-based-polymorphism
template <typename... ArgTs>
class PassManager {
 public:
    enum class VerificationLevelType {
        NONE     = 0, // do not perform any verification
        TOP_ONLY = 1, // execute postCheck() only in top-level passes
        FULL     = 2, // execute pre/postCheck() for all passes
    };

    PassManager()                       = default;
    PassManager(PassManager&&) noexcept = default;
    PassManager& operator=(PassManager&&) noexcept = default;

    /// @brief check capabilities all registered passes
    void capability_check(const nn_ir::NNIR& graph, ArgTs... args) {
        for (auto& pass : passes_) {
            pass->capability_check(graph, args...);
        }
    }

    /// @brief initialization all registered passes
    RetVal initialize(const UtilManager& util_manager, const TraitManager& trait_manager, ArgTs... args) {
        RetVal res = RetVal::SUCCESS;

        for (auto& pass : passes_) {
            pass->initialize(util_manager, trait_manager, args...);
        }

        return res;
    }

    /**
     * @brief.      Run pass
     * @returns.    enum RetVal success/failure
     */
    RetVal run(nn_ir::NNIR& graph, CompilationContext& ctx, ArgTs... args) {
        RetVal res = RetVal::SUCCESS;

        for (auto& pass : passes_) {
            res = pass->run(graph, ctx, args...);
            LOGE_IF(ME, res == RetVal::FAILURE, "pass run fail\n");
        }

        return res;
    }

    /**
     * @brief.      Run pass
     * @returns.    enum RetVal success/failure
     */
    RetVal run(nn_ir::NNIR& graph, CompilationContext& ctx, VerificationLevelType verify_level, ArgTs... args) {
        RetVal res = RetVal::SUCCESS;

        for (auto& pass : passes_) {
            if (verify_level == VerificationLevelType::FULL) {
                if (pass->preCheck(graph, args...) != RetVal::SUCCESS) {
                    Log::ME::E() << "Precondition of Pass violated";
                }
            }
            res = pass->run(graph, ctx, args...);
            LOGE_IF(ME, res == RetVal::FAILURE, "pass run fail\n");

            if (verify_level == VerificationLevelType::FULL) {
                if (pass->postCheck(graph, args...) != RetVal::SUCCESS) {
                    Log::ME::E() << "Postcondition of Pass violated";
                }
            }
        }

        return res;
    }

    /// @brief finalization all registered passes
    void finalize(ArgTs... args) {
        for (auto& pass : passes_) {
            pass->finalize(args...);
        }
    }

    /// @brief postCheck all registered passes
    RetVal postCheck(const nn_ir::NNIR& graph, ArgTs... args) {
        RetVal res = RetVal::SUCCESS;

        for (auto& pass : passes_) {
            auto pass_res = pass->postCheck(graph, args...);
            if (pass_res != RetVal::SUCCESS)
                res = pass_res;
        }

        return res;
    }

    /// @brief preCheck all registered passes
    RetVal preCheck(const nn_ir::NNIR& graph, ArgTs... args) const {
        RetVal res = RetVal::SUCCESS;

        for (auto& pass : passes_) {
            auto pass_res = pass->preCheck(graph, args...);
            if (pass_res != RetVal::SUCCESS)
                res = pass_res;
        }

        return res;
    }

    /**
     * @brief.      add pass
     * @details.    this function adds pass to passes_.
     */
    template <typename PassT>
    void addPass(PassT pass) {
        passes_.emplace_back(new internal::PassModel<PassT, ArgTs...>(std::move(pass)));
    }

    // Don't print anything from within wrappers
    static void trace(const char* name) {}

 private:
    std::vector<std::unique_ptr<internal::PassConcept<ArgTs...>>> passes_;
}; // class PassManager

} // namespace nn_compiler
