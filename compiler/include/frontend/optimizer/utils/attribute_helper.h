#pragma once

#include "ir/include/nn_network.h"

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

bool putAttributeInAtenAsTensor(layer_inID_type& layer_inID, dtensor_ptr_type& d_tensor);

bool putAttributeInAtenBatchNorm(layer_inID_type& layer_inID, dtensor_ptr_type& d_tensor);

bool putAttributeInAtenCat(layer_inID_type& layer_inID, dtensor_ptr_type& d_tensor);

bool putAttributeInAtenChunk(layer_inID_type& layer_inID, dtensor_ptr_type& d_tensor);

bool putAttributeInAtenClamp(layer_inID_type& layer_inID, dtensor_ptr_type& d_tensor);

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

bool putAttributeInAtenUnsqueeze(layer_inID_type& layer_inID, dtensor_ptr_type& d_tensor);

bool putAttributeInAtenWarn(layer_inID_type& layer_inID, dtensor_ptr_type& d_tensor);

bool putAttributeInPrimLoop(layer_inID_type& layer_inID, dtensor_ptr_type& d_tensor);

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

    AttributeHelper()
    {
        type_to_function_["aten::add"] = &putAttributeInAtenAdd;
        type_to_function_["aten::arange1"] = &putAttributeInAtenArange1;
        type_to_function_["aten::arange2"] = &putAttributeInAtenArange2;
        type_to_function_["aten::arange3"] = &putAttributeInAtenArange3;
        type_to_function_["aten::as_tensor"] = &putAttributeInAtenAsTensor;
        type_to_function_["aten::batch_norm"] = &putAttributeInAtenBatchNorm;
        type_to_function_["aten::cat"] = &putAttributeInAtenCat;
        type_to_function_["aten::chunk"] = &putAttributeInAtenChunk;
        type_to_function_["aten::clamp"] = &putAttributeInAtenClamp;
        type_to_function_["aten::contiguous"] = &putAttributeInAtenContiguous;
        type_to_function_["aten::conv2d"] = &putAttributeInAtenConv2d;
        type_to_function_["aten::__derive_index"] = &putAttributeInAtenDeriveIndex;
        type_to_function_["aten::dropout"] = &putAttributeInAtenDropout;
        type_to_function_["aten::dropout_"] = &putAttributeInAtenDropout;
        type_to_function_["aten::expand"] = &putAttributeInAtenExpand;
        type_to_function_["aten::embedding"] = &putAttributeInAtenEmbedding;
        type_to_function_["aten::format"] = &putAttributeInAtenFormat;
        type_to_function_["aten::gather"] = &putAttributeInAtenGather;
        type_to_function_["aten::__getitem__"] = &putAttributeInAtenGetItem;
        type_to_function_["aten::index_put_"] = &putAttributeInAtenIndexPut;
        type_to_function_["aten::index_select"] = &putAttributeInAtenIndexSelect;
        type_to_function_["aten::leaky_relu"] = &putAttributeInAtenLeakyRelu;
        type_to_function_["aten::log_softmax"] = &putAttributeInAtenLogSoftmax;
        type_to_function_["aten::lstm1"] = &putAttributeInAtenLstm1;
        type_to_function_["aten::lstm2"] = &putAttributeInAtenLstm2;
        type_to_function_["aten::max"] = &putAttributeInAtenMax;
        type_to_function_["aten::max_pool2d"] = &putAttributeInAtenMaxPool2d;
        type_to_function_["aten::min"] = &putAttributeInAtenMin;
        type_to_function_["aten::norm"] = &putAttributeInAtenNorm;
        type_to_function_["aten::ones"] = &putAttributeInAtenOnes;
        type_to_function_["aten::_pack_padded_sequence"] = &putAttributeInAtenPackPaddedSequence;
        type_to_function_["aten::_pad_packed_sequence"] = &putAttributeInAtenPadPackedSequence;
        type_to_function_["aten::select"] = &putAttributeInAtenSelect;
        type_to_function_["aten::_set_item"] = &putAttributeInAtenSetItem;
        type_to_function_["aten::size"] = &putAttributeInAtenSize;
        type_to_function_["aten::slice"] = &putAttributeInAtenSlice;
        type_to_function_["aten::softmax"] = &putAttributeInAtenSoftmax;
        type_to_function_["aten::squeeze"] = &putAttributeInAtenSqueeze;
        type_to_function_["aten::sub"] = &putAttributeInAtenSub;
        type_to_function_["aten::sum"] = &putAttributeInAtenSum;
        type_to_function_["aten::to1"] = &putAttributeInAtenTo1;
        type_to_function_["aten::to2"] = &putAttributeInAtenTo2;
        type_to_function_["aten::topk"] = &putAttributeInAtenTopk;
        type_to_function_["aten::transpose"] = &putAttributeInAtenTranspose;
        type_to_function_["aten::unsqueeze"] = &putAttributeInAtenUnsqueeze;
        type_to_function_["aten::unsqueeze_"] = &putAttributeInAtenUnsqueeze;
        type_to_function_["aten::warn"] = &putAttributeInAtenWarn;
        type_to_function_["prim::Loop"] = &putAttributeInPrimLoop;
        type_to_function_["prim::TupleIndex"] = &putAttributeInPrimTupleIndex;
    }

    bool putAttribute(std::string name, layer_inID_type& layer_inID, dtensor_ptr_type& d_tensor);

   private:
    std::map<std::string, putAttributeFunc> type_to_function_;
};

}  // namespace optimizer_utils
}  // namespace frontend
}  // namespace nn_compiler
