#pragma once

#include "common/include/pretty_print.hpp"
#include "ir/include/ir_types.hpp"
#include "compiler/include/middlend/trait/trait_manager.hpp"
#include "compiler/include/middlend/utils/util_manager.hpp"

#include <memory>

namespace nn_compiler {

class UtilManager;

/// @brief this class is mixin class for passes to reduce boilerplate code.
///        It should be used via CRTP and if needed can be extended
/// @note  this class is not intended as polymorphic class. It must not be used as follows:
/// @code
///        SomePass p;
///        PassMixin<SomePass> pm = &p;
///        pm->...
/// @endcode
template <typename PassT, typename... TArgs>
class PassMixin {
 public:
    static std::string getName() {
        static_assert(std::is_base_of<PassMixin, PassT>::value, "This class must be used via CRTP");
        return getTypeName<PassT>();
    }

    void capability_check(const nn_ir::NNIR& graph, TArgs... args) {}

    RetVal initialize(const UtilManager&, const TraitManager&, TArgs... args) { return RetVal::SUCCESS; }

    void finalize(TArgs... args) {}

    RetVal postCheck(const nn_ir::NNIR& graph, TArgs... args) { return RetVal::SUCCESS; }

    virtual RetVal preCheck(const nn_ir::NNIR& graph, TArgs... args) const { return RetVal::SUCCESS; }

    static void trace(const char* name) { Log::ME::I() << getName() << "::" << name << " is called"; }
};

/// @brief this class is used to clear all data of pass to call `clear()` method.
///        It should be used as follows:
///              RetVal run(nn_ir::NNIR& graph, CompilationContext& ctx, const ISystem& system) {
///                   PassRAII cleaner(cleaned_field1, cleaned_field2, /* etc */);
///              }
template <typename... ArgsT>
class PassRAII {
 public:
    explicit PassRAII(ArgsT&... args) : destroyed_objs_(args...) {}

    ~PassRAII() {
        std::apply([](ArgsT&... objs) { (objs.clear(), ...); }, destroyed_objs_);
    }

    PassRAII()                = delete;
    PassRAII(const PassRAII&) = delete;
    PassRAII(PassRAII&&)      = delete;
    PassRAII& operator=(const PassRAII&) = delete;
    PassRAII& operator=(PassRAII&&) = delete;

 private:
    std::tuple<ArgsT&...> destroyed_objs_;
};

} // namespace nn_compiler
