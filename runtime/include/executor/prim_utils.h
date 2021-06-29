#pragma once

#include <exception>
#include <vector>
#include <torch/script.h>
#include "../nnrt_types.h"

namespace nnrt
{
class NNRuntimeException : std::exception
{
   public:
    NNRuntimeException(std::string& msg) { this->msg = msg; }
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

torch::Tensor createPtTensor(void* data_ptr, std::vector<int64_t>& shape, DataType dtype);

}  // namespace nnrt
