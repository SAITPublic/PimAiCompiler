#include<torch/script.h>
#include<vector>
#include"executor/prim_utils.h"

namespace nnrt
{
torch::jit::IValue pop(std::vector<torch::jit::IValue>& stack)
{
    auto r = std::move(stack.back());
    stack.pop_back();
    return r;
}

void drop(std::vector<torch::jit::IValue>& stack, size_t n) { 
    stack.erase(stack.end() - n, stack.end()); 
}

}  // namespace nnrt
