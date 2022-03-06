#pragma once

#include <torch/script.h>
#include <exception>
#include <stack>
#include <vector>

#include "new_ir/include/tensors/data_tensor.h"
#include "new_ir/include/types.h"
#include "new_runtime/include/executor/utils.h"
// #include "new_runtime/include/types.h"

namespace nn_compiler
{
namespace runtime
{
using namespace nn_compiler::ir;
class NNRuntimeException : std::exception
{
    using std::exception::what;

   public:
    explicit NNRuntimeException(std::string msg) { this->msg = msg; }
    const char* what() { return msg.c_str(); }

   private:
    std::string msg;
};

torch::jit::IValue pop(std::vector<torch::jit::IValue>& stack);

void drop(std::vector<torch::jit::IValue>& stack, size_t n);

template <typename Type>
static inline void push_one(std::vector<torch::jit::IValue>& stack, Type&& arg)
{
    stack.emplace_back(std::forward<Type>(arg));
}

template <typename... Types>
void push(std::vector<torch::jit::IValue>& stack, Types&&... args)
{
    (void)std::initializer_list<int>{(push_one(stack, std::forward<Types>(args)), 0)...};
}

torch::Tensor createPtTensor(void* data_ptr, const std::vector<int64_t>& shape, DataType dtype,
                             const std::vector<int64_t>& stride = {});

std::vector<int64_t> getDataShapeFromSTensor(nn_compiler::ir::STensor& value);

std::vector<int64_t> getDataShapeFromVector(std::vector<int64_t>& value);

torch::Tensor loadTensor(const std::string& bin_file, const std::vector<int64_t>& shape, DataType dtype);

std::pair<int, DataType> parseNtype(std::string& ntype);

torch::jit::IValue convertVaraibleData2IValve(uint8_t* ptr, DataType d_type);

}  // namespace runtime
}  // namespace nn_compiler
