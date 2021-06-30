#include "executor/prim_utils.h"
#include <torch/script.h>
#include <stdexcept>
#include <vector>
#include "glog/logging.h"
#include "nnrt_types.h"

namespace nnrt
{
torch::jit::IValue pop(std::vector<torch::jit::IValue>& stack)
{
    auto r = std::move(stack.back());
    stack.pop_back();
    return r;
}

void drop(std::vector<torch::jit::IValue>& stack, size_t n) { stack.erase(stack.end() - n, stack.end()); }

/**
 * @brief Create a PyTorch Tensor
 *
 * @param data_ptr
 * @param shape
 * @param dtype
 * @return torch::Tensor
 */
torch::Tensor createPtTensor(void* data_ptr, std::vector<int64_t>& shape, DataType dtype)
{
    c10::ScalarType scalar_type;

    if (dtype == DataType::FLOAT32) {
        scalar_type = c10::ScalarType::Float;
    }
    else if (dtype == DataType::FLOAT16) {
        scalar_type = c10::ScalarType::Half;
    } else if (dtype == DataType::INT32) {
        scalar_type = c10::ScalarType::Int;
    } else {
        DLOG(ERROR) << "Unsupport dtype when create Tensor";
    }

    auto sizes = c10::IntArrayRef(shape);
    return torch::from_blob(data_ptr, sizes, c10::TensorOptions().dtype(scalar_type));
}

}  // namespace nnrt
