#include "executor/prim_ops.h"
#include <torch/script.h>
#include <string>
#include <unordered_map>
#include <vector>
#include "executor/prim_utils.h"
#include "glog/logging.h"
#include "stream_executor.h"
#include "executor/prim_utils.h"
#include "executor/prim_ops.h"


namespace nnrt
{
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


void primRaiseException(std::string msg) { throw nnrt::NNRuntimeException(msg); }

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
// tuple: c10::ivalue::Tuple
void primTupleConstruct(std::vector<torch::IValue>& stack, size_t num_inputs)
{
    std::vector<torch::jit::IValue> elems{std::make_move_iterator(stack.end() - num_inputs),
                                          std::make_move_iterator(stack.end())};
    nnrt::drop(stack, num_inputs);
    nnrt::push(stack, c10::ivalue::Tuple::create(std::move(elems)));
}


std::vector<torch::IValue> primTupleUnpack(c10::intrusive_ptr<c10::ivalue::Tuple> tuple)
{
    std::vector<torch::IValue> ret;
    for(auto& item : tuple->elements()){
        ret.push_back(item);
    }
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

/**
 * @brief The input to primIf is a cond with bool type
 * the caller should choose execution edge based on the retruned value
 *
 * @param cond
 * @return true/false
 */
bool primIf(bool cond) { return cond; }

/**
 * @brief primEndIf
 *  The GraphGen add the prim::endif Op at the end of PrimIf's block.
 *  Here's an example with primIf block, the input of primEndIf is %hx.3 or %hx0.2,
 *  the output of primEndIf is the reference of primEndIf's input
 *
 *  %hx.2 : (Tensor, Tensor) = prim::If(%214)
 *    block0():
 *       %zeros.2 : Tensor = aten::zeros(%364, %24, %58, %363, %58)
 *       %hx.3 : (Tensor, Tensor) = prim::TupleConstruct(%zeros.2, %zeros.2)
 *       -> (%hx.3)
 *    block1():
 *       %hx0.2 : (Tensor, Tensor) = prim::unchecked_cast(%state1.1)
 *       -> (%hx0.2)
 *  out = prim::endif(%hx0.2 or %hx.3)
 *
 * @tparam T
 * @param inputs
 * @return T
 */
// template <typename T>
// T primEndIf(T& inputs)
// {
//     auto ret = std::move<T>(inputs);
//     return ret;
// }


// STRING, DEVICE
std::string primStrConstsnt(void* data_ptr)
{
    std::string ret = static_cast<char*>(data_ptr);
    return ret;
}

torch::Tensor primTensorConstant(void* data_ptr, std::vector<int64_t>& shape, DataType dtype)
{
    return createPtTensor(data_ptr, shape, dtype);
}

std::vector<torch::Tensor> primEndLoop(const std::vector<torch::Tensor>& inputs) { return std::move(inputs); }

// blobs is a Global table
void primLoop(int max_trip_cnt, torch::Tensor& cond, int op_id, std::unordered_map<int, torch::Tensor>& blobs)
{
    // In Graphgen, Each PrimLoop Op contains a loop_index
    // example: for(loop_index = 0; loop_index < max_trip_cnt; loop_index++) { }
    OpNodeDescription* loop_index_node = getNextExecutionOp(new OpNodeDescription(op_id, "PrimLoop"));

    // Set initail value for loop_index, default is zero
    int loop_index = blobs.find(loop_index_node->id)->second.item().toInt();
    loop_index = 0;
    // modify the blob in blobs'table
    blobs[loop_index_node->id] = torch::tensor(loop_index);

    // skip LoopIndexNode
    OpNodeDescription* loop_start_op  = getNextExecutionOp(loop_index_node);
    OpNodeDescription* op_node = loop_start_op;
    
    // Generate a Loop
    while (loop_index < max_trip_cnt && cond.item().toBool()) {
        while (op_node->type != "PrimEndLoop") {
            // the cond may changed while execution
            executeOp(op_node);
            op_node = getNextExecutionOp(op_node);
        }
        loop_index++;
        blobs[loop_index_node->id] = torch::tensor(loop_index);
        // move to start
        op_node = loop_start_op;
    }
}

}  // namespace nnrt
