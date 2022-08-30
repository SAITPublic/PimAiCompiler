#pragma once

#include <torch/script.h>

#include "executor/utils/utils.h"
#include "ir/include/tensors/data_tensor.h"
#include "ir/include/types.h"

namespace nn_compiler
{
namespace runtime
{
namespace utils
{
using namespace nn_compiler::ir;

torch::Tensor createPtTensor(void* data_ptr, const std::vector<int64_t>& shape, DataType dtype,
                             const std::vector<int64_t>& stride = {});

std::vector<int64_t> getDataShapeFromSTensor(nn_compiler::ir::STensor& stensor);

torch::Tensor loadTensor(const std::string& bin_file, const std::vector<int64_t>& shape, DataType dtype);

std::pair<int, DataType> parseNtype(std::string& ntype);

torch::jit::IValue convertVaraibleData2IValve(uint8_t* ptr, DataType d_type);

class AUTO_Mutex
{
   private:
    static std::mutex mutex_;

   public:
    AUTO_Mutex();
    ~AUTO_Mutex();
    static void auto_lock() { mutex_.lock(); }
    static void auto_unlock() { mutex_.unlock(); }
};

}  // namespace utils
}  // namespace runtime
}  // namespace nn_compiler
