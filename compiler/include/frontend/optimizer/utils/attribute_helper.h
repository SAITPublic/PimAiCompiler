#pragma once

#include "ir/include/layers/all_layers.h"

namespace nn_compiler
{
namespace frontend
{
namespace optimizer_utils
{
typedef std::pair<const std::shared_ptr<nn_compiler::ir::NNLayer>, unsigned int> layer_inID_type;

typedef std::shared_ptr<nn_compiler::ir::DTensor> dtensor_ptr_type;

bool putAttributeInAtenAdd(layer_inID_type& layer_inID, dtensor_ptr_type& d_tensor);

bool putAttributeInAtenArange1(layer_inID_type& layer_inID, dtensor_ptr_type& d_tensor);

bool putAttributeInAtenArange2(layer_inID_type& layer_inID, dtensor_ptr_type& d_tensor);

bool putAttributeInAtenArange3(layer_inID_type& layer_inID, dtensor_ptr_type& d_tensor);

bool putAttributeInArgmax(layer_inID_type& layer_inID, dtensor_ptr_type& d_tensor);

bool putAttributeInAtenAsTensor(layer_inID_type& layer_inID, dtensor_ptr_type& d_tensor);

bool putAttributeInAtenBatchNorm(layer_inID_type& layer_inID, dtensor_ptr_type& d_tensor);

bool putAttributeInAtenCat(layer_inID_type& layer_inID, dtensor_ptr_type& d_tensor);

bool putAttributeInAtenChunk(layer_inID_type& layer_inID, dtensor_ptr_type& d_tensor);

bool putAttributeInAtenClamp(layer_inID_type& layer_inID, dtensor_ptr_type& d_tensor);

bool putAttributeInAtenClone(layer_inID_type& layer_inID, dtensor_ptr_type& d_tensor);

bool putAttributeInAtenContiguous(layer_inID_type& layer_inID, dtensor_ptr_type& d_tensor);

bool putAttributeInAtenConv2d(layer_inID_type& layer_inID, dtensor_ptr_type& d_tensor);

bool putAttributeInAtenDeriveIndex(layer_inID_type& layer_inID, dtensor_ptr_type& d_tensor);

bool putAttributeInAtenDropout(layer_inID_type& layer_inID, dtensor_ptr_type& d_tensor);

bool putAttributeInAtenExpand(layer_inID_type& layer_inID, dtensor_ptr_type& d_tensor);

bool putAttributeInAtenEmbedding(layer_inID_type& layer_inID, dtensor_ptr_type& d_tensor);

bool putAttributeInAtenFormat(layer_inID_type& layer_inID, dtensor_ptr_type& d_tensor);

bool putAttributeInAtenGather(layer_inID_type& layer_inID, dtensor_ptr_type& d_tensor);

bool putAttributeInAtenGetItem(layer_inID_type& layer_inID, dtensor_ptr_type& d_tensor);

bool putAttributeInAtenIndexPut(layer_inID_type& layer_inID, dtensor_ptr_type& d_tensor);

bool putAttributeInAtenIndexSelect(layer_inID_type& layer_inID, dtensor_ptr_type& d_tensor);

bool putAttributeInAtenLayerNorm(layer_inID_type& layer_inID, dtensor_ptr_type& d_tensor);

bool putAttributeInAtenLeakyRelu(layer_inID_type& layer_inID, dtensor_ptr_type& d_tensor);

bool putAttributeInAtenLogSoftmax(layer_inID_type& layer_inID, dtensor_ptr_type& d_tensor);

bool putAttributeInAtenLstm1(layer_inID_type& layer_inID, dtensor_ptr_type& d_tensor);

bool putAttributeInAtenLstm2(layer_inID_type& layer_inID, dtensor_ptr_type& d_tensor);

bool putAttributeInAtenMax(layer_inID_type& layer_inID, dtensor_ptr_type& d_tensor);

bool putAttributeInAtenMaxPool2d(layer_inID_type& layer_inID, dtensor_ptr_type& d_tensor);

bool putAttributeInAtenMin(layer_inID_type& layer_inID, dtensor_ptr_type& d_tensor);

bool putAttributeInAtenNorm(layer_inID_type& layer_inID, dtensor_ptr_type& d_tensor);

bool putAttributeInAtenOnes(layer_inID_type& layer_inID, dtensor_ptr_type& d_tensor);

bool putAttributeInAtenPackPaddedSequence(layer_inID_type& layer_inID, dtensor_ptr_type& d_tensor);

bool putAttributeInAtenPadPackedSequence(layer_inID_type& layer_inID, dtensor_ptr_type& d_tensor);

bool putAttributeInAtenSetItem(layer_inID_type& layer_inID, dtensor_ptr_type& d_tensor);

bool putAttributeInAtenSelect(layer_inID_type& layer_inID, dtensor_ptr_type& d_tensor);

bool putAttributeInAtenSize(layer_inID_type& layer_inID, dtensor_ptr_type& d_tensor);

bool putAttributeInAtenSlice(layer_inID_type& layer_inID, dtensor_ptr_type& d_tensor);

bool putAttributeInAtenSoftmax(layer_inID_type& layer_inID, dtensor_ptr_type& d_tensor);

bool putAttributeInAtenSqueeze(layer_inID_type& layer_inID, dtensor_ptr_type& d_tensor);

bool putAttributeInAtenSub(layer_inID_type& layer_inID, dtensor_ptr_type& d_tensor);

bool putAttributeInAtenSum(layer_inID_type& layer_inID, dtensor_ptr_type& d_tensor);

bool putAttributeInAtenTo1(layer_inID_type& layer_inID, dtensor_ptr_type& d_tensor);

bool putAttributeInAtenTo2(layer_inID_type& layer_inID, dtensor_ptr_type& d_tensor);

bool putAttributeInAtenTopk(layer_inID_type& layer_inID, dtensor_ptr_type& d_tensor);

bool putAttributeInAtenTranspose(layer_inID_type& layer_inID, dtensor_ptr_type& d_tensor);

bool putAttributeInAtenTriu(layer_inID_type& layer_inID, dtensor_ptr_type& d_tensor);

bool putAttributeInAtenUnsqueeze(layer_inID_type& layer_inID, dtensor_ptr_type& d_tensor);

bool putAttributeInAtenWarn(layer_inID_type& layer_inID, dtensor_ptr_type& d_tensor);

bool putAttributeInPrimLoop(layer_inID_type& layer_inID, dtensor_ptr_type& d_tensor);

bool putAttributeInPrimToList(layer_inID_type& layer_inID, dtensor_ptr_type& d_tensor);

bool putAttributeInPrimTupleIndex(layer_inID_type& layer_inID, dtensor_ptr_type& d_tensor);

template <typename T>
T getValueFromConstant(dtensor_ptr_type& d_tensor, nn_compiler::ir::LayerType layer_type, std::string attr_type)
{
    T ret_value;
    auto data = d_tensor->getData<T>();
    if ((*data).size() == 0) {
        DLOG(FATAL) << attr_type << " of " << convertLayerTypeToString(layer_type) << " cannot be NONE";
    } else {
        ret_value = (*data)[0];
    }
    return ret_value;
}

template <typename T>
std::vector<T> getVectorFromConstant(dtensor_ptr_type& d_tensor, nn_compiler::ir::LayerType layer_type,
                                     std::string attr_type)
{
    std::vector<T> ret_vec;
    auto data = d_tensor->getData<T>();
    if ((*data).size() == 0) {
        DLOG(FATAL) << attr_type << " of " << convertLayerTypeToString(layer_type) << " cannot be NONE";
    } else {
        for (auto item : *data) {
            ret_vec.push_back(item);
        }
    }
    return ret_vec;
}

std::string getStringFromConstant(dtensor_ptr_type& d_tensor, nn_compiler::ir::LayerType layer_type,
                                  std::string attr_type);

class AttributeHelper
{
   public:
    typedef bool (*putAttributeFunc)(layer_inID_type& layer_inID, dtensor_ptr_type& d_tensor);

    AttributeHelper();

    bool putAttribute(std::string name, layer_inID_type& layer_inID, dtensor_ptr_type& d_tensor);

   private:
    std::map<std::string, putAttributeFunc> type_to_function_;
};

}  // namespace optimizer_utils
}  // namespace frontend
}  // namespace nn_compiler
