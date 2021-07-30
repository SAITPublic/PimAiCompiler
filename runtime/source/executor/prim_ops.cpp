#include "executor/prim_ops.h"
#include <torch/script.h>
#include <string>
#include <unordered_map>
#include <vector>
#include "executor/prim_ops.h"
#include "executor/prim_utils.h"
#include "executor/stream_executor.h"
#include "glog/logging.h"

namespace nnrt
{
torch::Tensor primData(const torch::Tensor& input_tensor)
{
    auto ret = torch::autograd::Variable(input_tensor).variable_data();
    return ret;
}

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

template <typename T>
std::vector<T>& primEndLoop(const std::vector<T>& inputs)
{
    return inputs;
}

// STRING, DEVICE
std::string primStrConstsnt(void* data_ptr)
{
    std::string ret = static_cast<char*>(data_ptr);
    return ret;
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

void primRaiseException(std::string msg)
{
    nnrt::NNRuntimeException exception(msg);
    DLOG(INFO) << exception.what();
    throw exception;
}

torch::Tensor primTensorConstant(void* data_ptr, std::vector<int64_t>& shape, DataType dtype)
{
    return createPtTensor(data_ptr, shape, dtype);
}

// the unpacked tensors will append to stack
// torch::jit::Value can convert to tensor/int/bool/float and so-on
// tuple: c10::ivalue::Tuple
void primTupleConstruct(std::vector<torch::IValue>& stack, size_t num_inputs)
{
    std::vector<torch::jit::IValue> elems{std::make_move_iterator(stack.end() - num_inputs),
                                          std::make_move_iterator(stack.end())};
    nnrt::drop(stack, num_inputs);
    nnrt::push(stack, c10::ivalue::Tuple::create(std::move(elems)));
}

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

std::vector<torch::IValue> primTupleUnpack(c10::intrusive_ptr<c10::ivalue::Tuple> tuple)
{
    std::vector<torch::IValue> ret;
    for (auto& item : tuple->elements()) {
        ret.push_back(item);
    }
    return ret;
}

torch::IValue primUninitialized()
{
    auto ret = torch::IValue::uninitialized();
    return ret;
}
}  // namespace nnrt
