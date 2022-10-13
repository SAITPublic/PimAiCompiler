#include <torch/script.h>

#include "executor/op_executor/prim_ops.h"

namespace nn_compiler
{
namespace runtime
{
namespace op_executor
{
torch::Tensor primData(const torch::Tensor& input_tensor)
{
    auto ret = torch::autograd::Variable(input_tensor).variable_data();
    return ret;
}

c10::Device primDevice(const torch::Tensor& input_tensor)
{
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
    utils::drop(stack, num_inputs);
    utils::push(stack, std::move(vals));
}

void primListConstruct(std::vector<torch::IValue>& stack)
{
    if (stack[0].isTuple()) {
        auto items = c10::impl::GenericList(at::TupleType::create({}));
        items.reserve(stack.size());
        for (uint32_t idx = 0; idx < stack.size(); idx++) {
            items.emplace_back(stack.at(idx).toTuple());
        }
        utils::drop(stack, stack.size());
        utils::push(stack, std::move(items));
    } else {
        DLOG(FATAL) << "primListConstruct element type do not support!";
    }
}

void primListUnpack(std::vector<torch::IValue>& stack, size_t num_outputs)
{
    auto list = utils::pop(stack).toList();
    assert(list.size() == num_outputs);
    // insert the unpakced data
    stack.insert(stack.end(), list.begin(), list.end());
}

void primRaiseException(std::string msg)
{
    utils::NNRuntimeException exception(msg);
    DLOG(INFO) << exception.what();
    throw exception;
}

// the unpacked tensors will append to stack
// torch::jit::Value can convert to tensor/int/bool/float and so-on
// tuple: c10::ivalue::Tuple
void primTupleConstruct(std::vector<torch::IValue>& stack, size_t num_inputs)
{
    std::vector<torch::jit::IValue> elems{std::make_move_iterator(stack.end() - num_inputs),
                                          std::make_move_iterator(stack.end())};
    utils::drop(stack, num_inputs);
    utils::push(stack, c10::ivalue::Tuple::create(std::move(elems)));
}

torch::IValue primTupleIndex(const std::vector<torch::IValue>& inputs, int64_t index)
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

at::IValue primVariable(std::string ntype, std::vector<torch::IValue> inputs)
{
    std::stack<std::string> ntype_stack;
    std::stack<char> delimiter_stack;
    std::stack<int> element_num_stack;
    std::vector<torch::jit::IValue> iv;
    int pos = 0;
    int total_input_num = 0;
    for (uint32_t idx = 0; idx < ntype.length(); idx++) {
        if (ntype.at(idx) == '[' || ntype.at(idx) == ']' || ntype.at(idx) == ',') {
            if (idx - pos != 0) {
                ntype_stack.push(ntype.substr(pos, idx - pos));
            }

            pos = idx + 1;
            if (ntype.at(idx) == ',') {
                auto temp_num = element_num_stack.top() + 1;
                element_num_stack.pop();
                element_num_stack.push(temp_num);
            }
        }

        if (ntype.at(idx) == '[') {
            delimiter_stack.push('[');
            element_num_stack.push(1);
        } else if (ntype.at(idx) == ']') {
            delimiter_stack.pop();
            std::vector<torch::jit::IValue> temp_iv;
            at::ListTypePtr list_type = at::ListType::ofTensors();
            // list[[tensor, tensor], [scalar, scalar]]
            if (ntype_stack.top() != "List" && ntype_stack.top() != "Tuple") {
                for (uint32_t s_id = 0; s_id < element_num_stack.top(); s_id++) {
                    temp_iv.push_back(inputs.at(total_input_num + s_id));
                    ntype_stack.pop();
                }
                total_input_num += element_num_stack.top();
                element_num_stack.pop();

                // make temp_iv to tuple or list(the ivalue type in temp_iv must be not list or tuple)
                if (ntype_stack.top() == "List") {
                    ntype_stack.pop();
                    list_type = utils::inferTypeFromDataType(utils::inferDataType(temp_iv.at(0)));
                    primListConstruct(temp_iv, temp_iv.size(), list_type);  // list_type is what
                    iv.push_back(temp_iv.at(0));
                    temp_iv.clear();
                } else if (ntype_stack.top() == "Tuple") {
                    ntype_stack.pop();
                    primTupleConstruct(temp_iv, temp_iv.size());
                    iv.push_back(temp_iv.at(0));
                    temp_iv.clear();
                } else {
                    DLOG(FATAL) << "Variable ntype: " << ntype_stack.top() << "is wrong!";
                }
            } else {
                // list[tuple, tuple] tuple[tuple, tuple]
                auto temp_num = element_num_stack.top();
                element_num_stack.pop();
                int total_iv_num = 0;
                if (!element_num_stack.empty()) {
                    total_iv_num = element_num_stack.top() - 1;
                }
                element_num_stack.push(temp_num);
                for (uint32_t iv_id = 0; iv_id < element_num_stack.top(); iv_id++) {
                    temp_iv.push_back(iv.at(total_iv_num));
                    iv.erase(iv.begin() + total_iv_num);
                }
                element_num_stack.pop();
                if (ntype_stack.top() == "List") {
                    ntype_stack.pop();
                    primListConstruct(temp_iv);
                } else if (ntype_stack.top() == "Tuple") {
                    ntype_stack.pop();
                    primTupleConstruct(temp_iv, temp_iv.size());
                } else {
                    DLOG(FATAL) << "Variable ntype: " << ntype_stack.top() << "is wrong!";
                }

                // update output to iv
                iv.insert(iv.begin() + total_iv_num, temp_iv.at(0));
                temp_iv.clear();
            }
        }
    }
    return iv.at(0);
}

}  // namespace op_executor
}  // namespace runtime
}  // namespace nn_compiler
