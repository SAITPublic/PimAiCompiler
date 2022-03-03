#pragma once

#include <exception>
#include <stack>
#include <torch/script.h>
#include <vector>

#include "new_ir/include/types.h"
#include "new_runtime/include/types.h"
#include "new_runtime/include/executor/utils.h"

namespace nn_compiler
{
namespace runtime
{
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

torch::Tensor loadTensor(const std::string& bin_file, const std::vector<int64_t>& shape, DataType dtype);

std::pair<torch::jit::IValue, std::pair<int, DataType>> getVariableInfo(uint8_t* ptr,
                                                                        const std::string tensor_data_type,
                                                                        int total_size);
}  // namespace runtime
}  // namespace nn_compiler
