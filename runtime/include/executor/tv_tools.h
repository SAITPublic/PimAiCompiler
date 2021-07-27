#include <glog/logging.h>
#include <torch/script.h>
#include <string>
#include <unordered_map>
#include <vector>
#include "executor/prim_utils.h"

namespace nnrt
{
std::string showTensorInfo(const torch::Tensor& tensor)
{
    std::stringstream ss;
    ss << " Shape:" << tensor.sizes() << " Dim:" << tensor.dim() << " Dtype:" << tensor.dtype()
       << " Device:" << tensor.device();
    return ss.str();
}

bool checkTensorEqual(const torch::Tensor& tensor1, const torch::Tensor& tensor2)
{
    DLOG(INFO) << "tensor1:" << showTensorInfo(tensor1);
    DLOG(INFO) << "tensor2:" << showTensorInfo(tensor2);

    bool ret = tensor1.equal(tensor2);
    return ret;
}

class TVComparator
{
   public:
    ~TVComparator() { tv_.clear(); }
    TVComparator(const TVComparator&) = delete;
    TVComparator& operator=(const TVComparator&) = delete;
    static TVComparator& getInstance()
    {
        static TVComparator tv_comparator;
        return tv_comparator;
    }

   private:
    TVComparator() {}

   public:
    void loadTV(const std::string& file, const std::vector<int64_t>& shape, DataType dtype, const std::string& tag_name)
    {
        auto tensor = loadTensor(file, shape, dtype);
        tv_.insert({tag_name, tensor});
    }

    void clear() { tv_.clear(); }

    bool compare(const torch::Tensor& input_tensor, const std::string& cmp_name)
    {
        auto iter = tv_.find(cmp_name);
        assert(iter != tv_.end() && "Test vector not exist!");
        auto other_tensor = iter->second;
        bool status = checkTensorEqual(input_tensor, other_tensor);
        if (status) {
            DLOG(INFO) << "TVComparator: Success !";
        } else {
            DLOG(INFO) << "TVComparator: Failed !";
        }
        return status;
    }

   private:
    std::unordered_map<std::string, torch::Tensor> tv_;
};

}  // namespace nnrt
