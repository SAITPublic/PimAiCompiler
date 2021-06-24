#include <iostream>
#include <string>
#include <vector>
#include <torch/script.h>
#include "executor/prim_utils.h"


namespace nnrt {
    
c10::Device primDevice(const torch::Tensor& input_tensor)
{
    // int device_idx = input_tensor.get_device();
    c10::Device device = input_tensor.device();
    return device;
}

int64_t primDtype(const torch::Tensor& input_tensor)
{
    int64_t ret = static_cast<int64_t>(input_tensor.scalar_type());
    return ret;
}

torch::Tensor primData(const torch::Tensor& input_tensor)
{
    auto ret = torch::autograd::Variable(input_tensor).variable_data();
    return ret;
}

torch::IValue primUninitialized()
{
    auto ret = torch::IValue::uninitialized();
    return ret;
}

template <typename T>
T primUncheckedCast(const T& inputs)
{
    // no
    auto outputs = inputs;
    return outputs;
}

void primRaiseException(std::string& msg) { throw nnrt::NNRuntimeException(msg); }

torch::Tensor primTupleIndex(const std::vector<torch::Tensor>& inputs, int64_t index)
{
    // Convert an python index (which may be negative) into an index usable for a
    // C++ container
    auto normalizeIndex = [](int64_t idx, int64_t list_size) -> int64_t {
        if (idx < 0) {
            // Handle negative indexing
            idx = list_size + idx;
        }
        return idx;
    };
    int64_t norm_index = normalizeIndex(index, inputs.size());
    if (norm_index < 0 || norm_index > static_cast<int64_t>(inputs.size())) {
        throw std::out_of_range("Tuple list index out of range");
    }
    return inputs[norm_index];
}

// the unpacked tensors will append to stack
// torch::jit::Value can convert to tensor/int/bool/float and so-on
void primTupleConstruct(std::vector<torch::IValue>& stack, size_t num_inputs)
{
    std::vector<torch::jit::IValue> elems{std::make_move_iterator(stack.end() - num_inputs),
                                          std::make_move_iterator(stack.end())};
    nnrt::drop(stack, num_inputs);
    nnrt::push(stack, c10::ivalue::Tuple::create(std::move(elems)));
}

// the unpacked tensors will append to stack
void primTupleUnpack(std::vector<torch::IValue>& stack)
{
    auto tuple = nnrt::pop(stack).toTuple();
    stack.insert(stack.end(), tuple->elements().begin(), tuple->elements().end());
}

// For Tensor[], list_type=at::ListType::ofTensors()
void primListConstruct(std::vector<torch::IValue>& stack, size_t num_inputs, at::ListTypePtr type)
{
    c10::List<torch::jit::IValue> vals(type->getElementType());
    vals.reserve(num_inputs);
    for (size_t i = stack.size() - num_inputs; i < stack.size(); ++i) {
        vals.emplace_back(std::move(stack[i]));
    }
    nnrt::drop(stack, num_inputs);
    nnrt::push(stack, std::move(vals));
}

void primListUnpack(std::vector<torch::IValue>& stack, size_t num_outputs)
{
    auto list = nnrt::pop(stack).toList();
    assert(list.size() == num_outputs);
    // insert the unpakced data
    stack.insert(stack.end(), list.begin(), list.end());
}

}  // namespace nnrt
