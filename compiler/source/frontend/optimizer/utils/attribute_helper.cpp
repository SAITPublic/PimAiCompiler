#include "frontend/optimizer/utils/attribute_helper.h"

namespace nn_compiler
{
namespace frontend
{
namespace optimizer_utils
{
typedef std::pair<const std::shared_ptr<nn_compiler::ir::NNLayer>, unsigned int> layer_inID_type;

typedef std::shared_ptr<nn_compiler::ir::DTensor> dtensor_ptr_type;

bool putAttributeInAtenAdd(layer_inID_type& layer_inID, dtensor_ptr_type& d_tensor)
{
    auto cur_layer = std::dynamic_pointer_cast<ir::AtenAddLayer>(layer_inID.first);
    auto idx = layer_inID.second;

    // e.g.
    // (1) %symbols_added0.1 : int = aten::add(%symbols_added.3, %39)
    //     %symbols_added.3 and % 39 are inputs;
    // (2) 480 : Tensor,  = aten::add (473, 443, 59, )
    //     473 and 443 are inputs, 59 is attribute: alpha.
    if (idx == 0 || idx == 1) {
        DLOG(INFO) << "prim::Constant attempts to set into the input of layer: " << layer_inID.first->getName();
        return false;
    } else if (idx == 2) {
        cur_layer->setAlpha(getValueFromConstant<int64_t>(d_tensor, cur_layer->getType(), "alpha"));
    } else {
        DLOG(FATAL) << "Incorrect data from prim::Constant";
    }
    return true;
}

bool putAttributeInAtenArange1(layer_inID_type& layer_inID, dtensor_ptr_type& d_tensor)
{
    auto cur_layer = std::dynamic_pointer_cast<ir::AtenArange1Layer>(layer_inID.first);
    auto idx = layer_inID.second;
    auto layer_type = cur_layer->getType();

    // e.g.  Tensor = aten::arange (295, 55, 15, 296, 15, )
    //  inputs are attr: end, dtype, layout, device, pin_memory seperately.
    if (idx == 0) {
        cur_layer->setEnd(getValueFromConstant<int64_t>(d_tensor, layer_type, "end"));
    } else if (idx == 1) {
        auto data = d_tensor->getData<int64_t>();
        if ((*data).size()) {
            cur_layer->setDtype((*data)[0]);
        } else {
            DLOG(INFO) << "dtype of aten::arange is set to NONE";
            return false;
        }
    } else if (idx == 2) {
        auto data = d_tensor->getData<int64_t>();
        if ((*data).size()) {
            cur_layer->setLayout((*data)[0]);
        } else {
            DLOG(INFO) << "layout of aten::arange is set to NONE";
            return false;
        }
    } else if (idx == 3) {
        cur_layer->setDevice(getStringFromConstant(d_tensor, cur_layer->getType(), "device"));
    } else if (idx == 4) {
        auto data = d_tensor->getData<int>();
        if ((*data).size()) {
            cur_layer->setPinMemory((*data)[0] != 0);
        } else {
            DLOG(INFO) << "pin_memory of aten::arange is set to NONE";
            return false;
        }
    } else {
        DLOG(FATAL) << "Incorrect data from prim::Constant";
    }
    return true;
}

bool putAttributeInAtenArange2(layer_inID_type& layer_inID, dtensor_ptr_type& d_tensor)
{
    auto cur_layer = std::dynamic_pointer_cast<ir::AtenArange2Layer>(layer_inID.first);
    auto idx = layer_inID.second;
    auto layer_type = cur_layer->getType();

    // e.g.  Tensor = aten::arange (54, 295, 55, 15, 296, 15, )
    //  inputs are attr: start, end, dtype, layout, device, pin_memory seperately.
    if (idx == 0) {
        cur_layer->setStart(getValueFromConstant<int64_t>(d_tensor, layer_type, "start"));
    } else if (idx == 1) {
        cur_layer->setEnd(getValueFromConstant<int64_t>(d_tensor, layer_type, "end"));
    } else if (idx == 2) {
        auto data = d_tensor->getData<int64_t>();
        if ((*data).size()) {
            cur_layer->setDtype((*data)[0]);
        } else {
            DLOG(INFO) << "dtype of aten::arange is set to NONE";
            return false;
        }
    } else if (idx == 3) {
        auto data = d_tensor->getData<int64_t>();
        if ((*data).size()) {
            cur_layer->setLayout((*data)[0]);
        } else {
            DLOG(INFO) << "layout of aten::arange is set to NONE";
            return false;
        }
    } else if (idx == 4) {
        cur_layer->setDevice(getStringFromConstant(d_tensor, cur_layer->getType(), "device"));
    } else if (idx == 5) {
        auto data = d_tensor->getData<int>();
        if ((*data).size()) {
            cur_layer->setPinMemory((*data)[0] != 0);
        } else {
            DLOG(INFO) << "pin_memory of aten::arange is set to NONE";
            return false;
        }
    } else {
        DLOG(FATAL) << "Incorrect data from prim::Constant";
    }
    return true;
}

bool putAttributeInAtenArange3(layer_inID_type& layer_inID, dtensor_ptr_type& d_tensor)
{
    auto cur_layer = std::dynamic_pointer_cast<ir::AtenArange3Layer>(layer_inID.first);
    auto idx = layer_inID.second;
    auto layer_type = cur_layer->getType();

    // e.g.  Tensor = aten::arange (54, 183, 64, 55, 15, 296, 15, )
    //  inputs are attr: start, end, step, dtype, layout, device, pin_memory seperately.
    if (idx == 0) {
        cur_layer->setStart(getValueFromConstant<int64_t>(d_tensor, layer_type, "start"));
    } else if (idx == 1) {
        cur_layer->setEnd(getValueFromConstant<int64_t>(d_tensor, layer_type, "end"));
    } else if (idx == 2) {
        cur_layer->setStep(getValueFromConstant<int64_t>(d_tensor, layer_type, "step"));
    } else if (idx == 3) {
        auto data = d_tensor->getData<int64_t>();
        if ((*data).size()) {
            cur_layer->setDtype((*data)[0]);
        } else {
            DLOG(INFO) << "dtype of aten::arange is set to NONE";
            return false;
        }
    } else if (idx == 4) {
        auto data = d_tensor->getData<int64_t>();
        if ((*data).size()) {
            cur_layer->setLayout((*data)[0]);
        } else {
            DLOG(INFO) << "layout of aten::arange is set to NONE";
            return false;
        }
    } else if (idx == 5) {
        cur_layer->setDevice(getStringFromConstant(d_tensor, cur_layer->getType(), "device"));
    } else if (idx == 6) {
        auto data = d_tensor->getData<int>();
        if ((*data).size()) {
            cur_layer->setPinMemory((*data)[0] != 0);
        } else {
            DLOG(INFO) << "pin_memory of aten::arange is set to NONE";
            return false;
        }
    } else {
        DLOG(FATAL) << "Incorrect data from prim::Constant";
    }
    return true;
}

bool putAttributeInAtenAsTensor(layer_inID_type& layer_inID, dtensor_ptr_type& d_tensor)
{
    auto cur_layer = std::dynamic_pointer_cast<ir::AtenAsTensorLayer>(layer_inID.first);
    auto idx = layer_inID.second;
    auto layer_type = cur_layer->getType();
    // e.g. Tensor = aten::as_tensor(%55, %56, %57)
    // %55 is input, %56, %57 is attrivute:dtype,device.
    if (idx == 0) {
        DLOG(INFO) << "prim::Constant attempts to set into the input of layer: " << layer_inID.first->getName();
        return false;
    } else if (idx == 1) {
        auto data = d_tensor->getData<int64_t>();
        if ((*data).size()) {
            cur_layer->setDtype((*data)[0]);
        } else {
            DLOG(INFO) << "dtype of aten::as_tensor is set to NONE";
            return false;
        }
    } else if (idx == 2) {
        cur_layer->setDevice(getStringFromConstant(d_tensor, cur_layer->getType(), "device"));
    } else {
        DLOG(FATAL) << "Incorrect data from prim::Constant";
    }
    return true;
}

bool putAttributeInAtenBatchNorm(layer_inID_type& layer_inID, dtensor_ptr_type& d_tensor)
{
    auto cur_layer = std::dynamic_pointer_cast<ir::AtenBatchNorm2dLayer>(layer_inID.first);
    auto idx = layer_inID.second;

    if (idx == 0) {
        DLOG(INFO) << "prim::Constant attempts to set into the input of layer: " << layer_inID.first->getName();
        return false;
    } else if (idx == 1) {
        DLOG(INFO) << "weights of aten::batch_norm have been set in layer builder stage";
    } else if (idx == 2) {
        DLOG(INFO) << "bias of aten::batch_norm have been set in layer builder stage";
    } else if (idx == 3) {
        DLOG(INFO) << "running_mean of aten::batch_norm have not been set in layer builder stage, don't delete";
        return false;
    } else if (idx == 4) {
        DLOG(INFO) << "running_var of aten::batch_norm have not been set in layer builder stage, don't delete";
        return false;
    } else if (idx == 5) {
        cur_layer->setTraining(getValueFromConstant<int>(d_tensor, cur_layer->getType(), "training"));
    } else if (idx == 6) {
        cur_layer->setMomentum(getValueFromConstant<double>(d_tensor, cur_layer->getType(), "momentum"));
    } else if (idx == 7) {
        cur_layer->setEps(getValueFromConstant<double>(d_tensor, cur_layer->getType(), "eps"));
    } else if (idx == 8) {
        cur_layer->setCudnnEnabled(getValueFromConstant<int>(d_tensor, cur_layer->getType(), "cudnn_enabled"));
    } else {
        DLOG(FATAL) << "Incorrect data from prim::Constant";
    }
    return true;
}

bool putAttributeInAtenCat(layer_inID_type& layer_inID, dtensor_ptr_type& d_tensor)
{
    auto cur_layer = std::dynamic_pointer_cast<ir::AtenCatLayer>(layer_inID.first);
    auto idx = layer_inID.second;

    // e.g. Tensor = aten::cat(%seq.1, %24) , %seq.1 is input, %24 is attribute: dim.
    if (idx == 0) {
        DLOG(INFO) << "prim::Constant attempts to set into the input of layer: " << layer_inID.first->getName();
        return false;
    } else if (idx == 1) {
        cur_layer->setDim(getValueFromConstant<int>(d_tensor, cur_layer->getType(), "dim"));
    } else {
        DLOG(FATAL) << "Incorrect data from prim::Constant";
    }
    return true;
}

bool putAttributeInAtenChunk(layer_inID_type& layer_inID, dtensor_ptr_type& d_tensor)
{
    auto cur_layer = std::dynamic_pointer_cast<ir::AtenChunkLayer>(layer_inID.first);
    auto idx = layer_inID.second;
    auto layer_type = cur_layer->getType();

    // e.g. Tensor[] = aten::chunk(%hidden.1, %56, %counter.1)
    // %hidden.1 is input, %56, %counter.1 is attrivute:chunks,dim.
    if (idx == 0) {
        DLOG(INFO) << "prim::Constant attempts to set into the input of layer: " << layer_inID.first->getName();
        return false;
    } else if (idx == 1) {
        cur_layer->setChunks(getValueFromConstant<int>(d_tensor, layer_type, "chunks"));
    } else if (idx == 2) {
        cur_layer->setDim(getValueFromConstant<int>(d_tensor, layer_type, "dim"));
    } else {
        DLOG(FATAL) << "Incorrect data from prim::Constant";
    }
    return true;
}

bool putAttributeInAtenClamp(layer_inID_type& layer_inID, dtensor_ptr_type& d_tensor)
{
    auto cur_layer = std::dynamic_pointer_cast<ir::AtenClampLayer>(layer_inID.first);
    auto idx = layer_inID.second;
    auto layer_type = cur_layer->getType();

    // e.g. Tensor = aten::clamp(%penalty.1, %counter.1, %149)
    // %penalty.1 is input, %counter.1, %149 is attrivute:min,max.
    if (idx == 0) {
        DLOG(INFO) << "prim::Constant attempts to set into the input of layer: " << layer_inID.first->getName();
        return false;
    } else if (idx == 1) {
        cur_layer->setMin(getValueFromConstant<int>(d_tensor, layer_type, "min"));
    } else if (idx == 2) {
        cur_layer->setMax(getValueFromConstant<int>(d_tensor, layer_type, "max"));
    } else {
        DLOG(FATAL) << "Incorrect data from prim::Constant";
    }
    return true;
}

bool putAttributeInAtenClone(layer_inID_type& layer_inID, dtensor_ptr_type& d_tensor)
{
    auto cur_layer = std::dynamic_pointer_cast<ir::AtenCloneLayer>(layer_inID.first);
    auto idx = layer_inID.second;

    // e.g. Tensor = aten::clone(tensor, %memory_format)
    // tensor is input, %memory_format is attrivute:memory_format.
    if (idx == 0) {
        DLOG(INFO) << "prim::Constant attempts to set into the input of layer: " << layer_inID.first->getName();
        return false;
    } else if (idx == 1) {
        cur_layer->setMemoryFormat(getValueFromConstant<int>(d_tensor, cur_layer->getType(), "memory_format"));
    } else {
        DLOG(FATAL) << "Incorrect data from prim::Constant";
    }
    return true;
}

bool putAttributeInAtenContiguous(layer_inID_type& layer_inID, dtensor_ptr_type& d_tensor)
{
    auto cur_layer = std::dynamic_pointer_cast<ir::AtenContiguousLayer>(layer_inID.first);
    auto idx = layer_inID.second;

    // e.g. Tensor = aten::contiguous(tensor, %counter.1)
    // tensor is input, %counter.1 is attrivute:memory_format.
    if (idx == 0) {
        DLOG(INFO) << "prim::Constant attempts to set into the input of layer: " << layer_inID.first->getName();
        return false;
    } else if (idx == 1) {
        cur_layer->setMemoryFormat(getValueFromConstant<int>(d_tensor, cur_layer->getType(), "memory_format"));
    } else {
        DLOG(FATAL) << "Incorrect data from prim::Constant";
    }
    return true;
}

bool putAttributeInAtenConv2d(layer_inID_type& layer_inID, dtensor_ptr_type& d_tensor)
{
    auto cur_layer = std::dynamic_pointer_cast<ir::AtenConv2dLayer>(layer_inID.first);
    auto idx = layer_inID.second;
    auto layer_type = cur_layer->getType();

    // e.g. Tensor = aten::conv2d(%x.1, %49, %50, %52, %53, %54, %51)
    // Tensor conv2d(const Tensor & input, const Tensor & weight, const Tensor & bias,
    //               IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation,
    //               int64_t groups);
    // %x.1 is input, %49, %50 is weights and bias.
    // attribute:%52:stride, %53:padding, %54:dialation, %51:groups
    if (idx == 0) {
        DLOG(INFO) << "prim::Constant attempts to set into the input of layer: " << layer_inID.first->getName();
        return false;
    } else if (idx == 1) {
        DLOG(INFO) << "weights of aten::conv2d have been set in layer builder stage";
    } else if (idx == 2) {
        DLOG(INFO) << "bias of aten::conv2d have been set in layer builder stage";
    } else if (idx == 3) {
        cur_layer->setStride(getVectorFromConstant<int64_t>(d_tensor, layer_type, "stride"));
    } else if (idx == 4) {
        cur_layer->setPadding(getVectorFromConstant<int64_t>(d_tensor, layer_type, "padding"));
    } else if (idx == 5) {
        cur_layer->setDialation(getVectorFromConstant<int64_t>(d_tensor, layer_type, "dialation"));
    } else if (idx == 6) {
        cur_layer->setGroups(getValueFromConstant<int64_t>(d_tensor, layer_type, "groups"));
    } else {
        DLOG(FATAL) << "Incorrect data from prim::Constant";
    }
    return true;
}

bool putAttributeInAtenDeriveIndex(layer_inID_type& layer_inID, dtensor_ptr_type& d_tensor)
{
    auto cur_layer = std::dynamic_pointer_cast<ir::AtenDeriveIndexLayer>(layer_inID.first);
    auto idx = layer_inID.second;
    auto layer_type = cur_layer->getType();

    // e.g. int = aten::__derive_index(%9, %5, %5)
    // %5 is attribute: index and step respectively.
    if (idx == 0) {
        DLOG(INFO) << "prim::Constant attempts to set into the input of layer: " << layer_inID.first->getName();
        return false;
    } else if (idx == 1) {
        cur_layer->setStart(getValueFromConstant<int64_t>(d_tensor, layer_type, "start"));
    } else if (idx == 2) {
        cur_layer->setStep(getValueFromConstant<int64_t>(d_tensor, layer_type, "step"));
    } else {
        DLOG(FATAL) << "Incorrect data from prim::Constant";
    }
    return true;
}

bool putAttributeInAtenDropout(layer_inID_type& layer_inID, dtensor_ptr_type& d_tensor)
{
    auto cur_layer = std::dynamic_pointer_cast<ir::AtenDropoutLayer>(layer_inID.first);
    auto idx = layer_inID.second;
    auto layer_type = cur_layer->getType();

    // e.g. Tensor = aten::dropout(%x2.1, %251, %51), %x2.1 is input,
    // %251 is attribute: proportion, %51 is attribute: train.
    if (idx == 0) {
        DLOG(INFO) << "prim::Constant attempts to set into the input of layer: " << layer_inID.first->getName();
        return false;
    } else if (idx == 1) {
        cur_layer->setProportion(getValueFromConstant<double>(d_tensor, layer_type, "proportion"));
    } else if (idx == 2) {
        cur_layer->setTrain(getValueFromConstant<int>(d_tensor, layer_type, "train"));
    } else {
        DLOG(FATAL) << "Incorrect data from prim::Constant";
    }
    return true;
}

bool putAttributeInAtenEmbedding(layer_inID_type& layer_inID, dtensor_ptr_type& d_tensor)
{
    auto cur_layer = std::dynamic_pointer_cast<ir::AtenEmbeddingLayer>(layer_inID.first);
    auto idx = layer_inID.second;
    auto layer_type = cur_layer->getType();

    // e.g. Tensor = aten::embedding(%self.embed.weight, %y.6, %padding_idx.19,
    //                               %self.dec_rnn.lstm.training, %self.dec_rnn.lstm.training)
    // %self.embed.weight is attribute weights, %y.6 is input.
    // %padding_idx.19 is attribute: padding_idx
    // %self.dec_rnn.lstm.training is value of attributes: scale_grad_by_freq and sparse.
    if (idx == 0) {
        DLOG(INFO) << "weights of aten::embedding will be processed in later pass.";
        return false;
    } else if (idx == 1) {
        DLOG(INFO) << "prim::Constant attempts to set into the input of layer: " << layer_inID.first->getName();
        return false;
    } else if (idx == 2) {
        cur_layer->setPaddingIdx(getValueFromConstant<int64_t>(d_tensor, layer_type, "padding_idx"));
    } else if (idx == 3) {
        cur_layer->setScaleGrad(getValueFromConstant<int64_t>(d_tensor, layer_type, "scale_grad_by_freq"));
    } else if (idx == 4) {
        cur_layer->setSparse(getValueFromConstant<int>(d_tensor, layer_type, "sparse"));
    } else {
        DLOG(FATAL) << "Incorrect data from prim::Constant";
    }
    return true;
}

bool putAttributeInAtenExpand(layer_inID_type& layer_inID, dtensor_ptr_type& d_tensor)
{
    auto cur_layer = std::dynamic_pointer_cast<ir::AtenExpandLayer>(layer_inID.first);
    auto idx = layer_inID.second;

    // e.g. Tensor = aten::expand(%f.4, %16, %self.net.2.training)
    // %f.4 and %16 are inputs, %self.net.2.training is attribute: implicit.
    if (idx == 0 || idx == 1) {
        DLOG(INFO) << "prim::Constant attempts to set into the input of layer: " << layer_inID.first->getName();
        return false;
    } else if (idx == 2) {
        cur_layer->setImplicit(getValueFromConstant<int>(d_tensor, cur_layer->getType(), "implicit"));
    } else {
        DLOG(FATAL) << "Incorrect data from prim::Constant";
    }
    return true;
}

bool putAttributeInAtenFormat(layer_inID_type& layer_inID, dtensor_ptr_type& d_tensor)
{
    auto cur_layer = std::dynamic_pointer_cast<ir::AtenFormatLayer>(layer_inID.first);
    auto idx = layer_inID.second;

    // e.g. str = aten::format(%30, %31, %47)
    // %31 and %47 are inputs, %30 is attribute: assembly_format
    if (idx == 1 || idx == 2) {
        DLOG(INFO) << "prim::Constant attempts to set into the input of layer: " << layer_inID.first->getName();
        return false;
    } else if (idx == 0) {
        cur_layer->setAssemblyFormat(getStringFromConstant(d_tensor, cur_layer->getType(), "assembly_format"));
    } else {
        DLOG(FATAL) << "Incorrect data from prim::Constant";
    }
    return true;
}

bool putAttributeInAtenGather(layer_inID_type& layer_inID, dtensor_ptr_type& d_tensor)
{
    auto cur_layer = std::dynamic_pointer_cast<ir::AtenGatherLayer>(layer_inID.first);
    auto idx = layer_inID.second;
    auto layer_type = cur_layer->getType();

    // e.g. Tensor = aten::gather(tensor, %149, tensor, %51)
    // tensor is inputs, %149, %51 is attribute: dim, sparse_grad.
    if (idx == 0 || idx == 2) {
        DLOG(INFO) << "prim::Constant attempts to set into the input of layer: " << layer_inID.first->getName();
        return false;
    } else if (idx == 1) {
        cur_layer->setDim(getValueFromConstant<int>(d_tensor, layer_type, "dim"));
    } else if (idx == 3) {
        cur_layer->setSparseGrad(getValueFromConstant<int>(d_tensor, layer_type, "sparse_grad"));
    } else {
        DLOG(FATAL) << "Incorrect data from prim::Constant";
    }
    return true;
}

bool putAttributeInAtenGetItem(layer_inID_type& layer_inID, dtensor_ptr_type& d_tensor)
{
    auto cur_layer = std::dynamic_pointer_cast<ir::AtenGetItemLayer>(layer_inID.first);
    auto idx = layer_inID.second;

    // e.g. %375 : int = aten::__getitem__(%sentence.1, %69)
    // %sentence.1 is inputs, %69 is attribute: idx.
    if (idx == 0) {
        DLOG(INFO) << "prim::Constant attempts to set into the input of layer: " << layer_inID.first->getName();
        return false;
    } else if (idx == 1) {
        cur_layer->setIdx(getValueFromConstant<int>(d_tensor, cur_layer->getType(), "idx"));
    } else {
        DLOG(FATAL) << "Incorrect data from prim::Constant";
    }
    return true;
}

bool putAttributeInAtenIndexPut(layer_inID_type& layer_inID, dtensor_ptr_type& d_tensor)
{
    auto cur_layer = std::dynamic_pointer_cast<ir::AtenIndexPutLayer>(layer_inID.first);
    auto idx = layer_inID.second;

    // e.g. Tensor = aten::index_put_(tensor,tensor, tensor, %51)
    // tensor,tensor, tensor are input, %51 is attribute: accumulate.
    if (idx == 0 || idx == 1 || idx == 2) {
        DLOG(INFO) << "prim::Constant attempts to set into the input of layer: " << layer_inID.first->getName();
        return false;
    } else if (idx == 3) {
        cur_layer->setAccumulate(getValueFromConstant<int>(d_tensor, cur_layer->getType(), "accumulate"));
    } else {
        DLOG(FATAL) << "Incorrect data from prim::Constant";
    }
    return true;
}

bool putAttributeInAtenIndexSelect(layer_inID_type& layer_inID, dtensor_ptr_type& d_tensor)
{
    auto cur_layer = std::dynamic_pointer_cast<ir::AtenIndexSelectLayer>(layer_inID.first);
    auto idx = layer_inID.second;

    // e.g. Tensor = aten::index_select(%1747, %counter.1, %mask1.1)
    // %1747, %mask1.1 are input, %counter.1 is attribute: dim.
    if (idx == 0 || idx == 2) {
        DLOG(INFO) << "prim::Constant attempts to set into the input of layer: " << layer_inID.first->getName();
        return false;
    } else if (idx == 1) {
        cur_layer->setDim(getValueFromConstant<int>(d_tensor, cur_layer->getType(), "dim"));
    } else {
        DLOG(FATAL) << "Incorrect data from prim::Constant";
    }
    return true;
}

bool putAttributeInAtenLayerNorm(layer_inID_type& layer_inID, dtensor_ptr_type& d_tensor)
{
    auto cur_layer = std::dynamic_pointer_cast<ir::AtenLayerNormLayer>(layer_inID.first);
    auto idx = layer_inID.second;
    auto layer_type = cur_layer->getType();

    // e.g.  %enc_output.23 : Tensor = aten::layer_norm
    //           (%54, %3317, %self.model.encoder.layer_norm.weight, %self.model.encoder.layer_norm.bias, %6, %18)
    // %54 is input, %3317 is normalized_shape, 
    // %self.model.encoder.layer_norm.weight is weight,  %self.model.encoder.layer_norm.bias is bias,
    // %6 is eps, %18 is cudnn_enable
    if (idx == 0) {
        DLOG(INFO) << "prim::Constant attempts to set into the input of layer: " << layer_inID.first->getName();
        return false;
    } else if (idx == 1) {
        cur_layer->setNormalizedShape(getVectorFromConstant<int>(d_tensor, layer_type, "normalized_shape"));
    } else if (idx == 2) {
        DLOG(INFO) << "weights of aten::layer_norm have been set in layer builder stage";
    } else if (idx == 3) {
        DLOG(INFO) << "bias of aten::layer_norm have been set in layer builder stage";
    } else if (idx == 4) {
        cur_layer->setEps(getValueFromConstant<float>(d_tensor, layer_type, "eps"));
    } else if (idx == 5) {
        cur_layer->setCudnnEnable(getValueFromConstant<int>(d_tensor, layer_type, "cudnn_enable"));
    } else {
        DLOG(FATAL) << "Incorrect data from prim::Constant";
    }
    return true;
}

bool putAttributeInAtenLeakyRelu(layer_inID_type& layer_inID, dtensor_ptr_type& d_tensor)
{
    auto cur_layer = std::dynamic_pointer_cast<ir::AtenLeakyReluLayer>(layer_inID.first);
    auto idx = layer_inID.second;

    // e.g.  Tensor = aten::leaky_relu(%x0.1, %57)
    // %x0.1 is input, %57 is attribute: scalar.
    if (idx == 0) {
        DLOG(INFO) << "prim::Constant attempts to set into the input of layer: " << layer_inID.first->getName();
        return false;
    } else if (idx == 1) {
        cur_layer->setScalar(getValueFromConstant<double>(d_tensor, cur_layer->getType(), "scalar"));
    } else {
        DLOG(FATAL) << "Incorrect data from prim::Constant";
    }
    return true;
}

bool putAttributeInAtenLogSoftmax(layer_inID_type& layer_inID, dtensor_ptr_type& d_tensor)
{
    auto cur_layer = std::dynamic_pointer_cast<ir::AtenLogSoftmaxLayer>(layer_inID.first);
    auto idx = layer_inID.second;
    auto layer_type = cur_layer->getType();

    // e.g. Tensor = aten::log_softmax(%x23.1, %117, %57)
    // %x23.1 is input, %117, %57 is attribute: dim, dtype.
    if (idx == 0) {
        DLOG(INFO) << "prim::Constant attempts to set into the input of layer: " << layer_inID.first->getName();
        return false;
    } else if (idx == 1) {
        cur_layer->setDim(getValueFromConstant<int64_t>(d_tensor, layer_type, "dim"));
    } else if (idx == 2) {
        auto data = d_tensor->getData<int64_t>();
        if ((*data).size()) {
            cur_layer->setDType((*data)[0]);
        } else {
            DLOG(INFO) << "dtype of aten::log_softmax is set to NONE";
            return false;
        }
    } else {
        DLOG(FATAL) << "Incorrect data from prim::Constant";
    }
    return true;
}

bool putAttributeInAtenLstm1(layer_inID_type& layer_inID, dtensor_ptr_type& d_tensor)
{
    auto cur_layer = std::dynamic_pointer_cast<ir::AtenLSTM1Layer>(layer_inID.first);
    auto idx = layer_inID.second;
    auto layer_type = cur_layer->getType();

    // e.g. Tensor, %81 : Tensor, %82 : Tensor = aten::lstm(%x_padded.1, %79, %self.pre_rnn.lstm._flat_weights,
    //                                                      %22, %24, %21, %self.pre_rnn.lstm.training,
    //                                                      %self.pre_rnn.lstm.training,
    //                                                      %self.pre_rnn.lstm.training)
    // %x_padded.1 and %79 are inputs, %self.pre_rnn.lstm._flat_weights is weight,
    // %22 is attribute: has_biases, %24 is attribute: _num_layers, %21 is attribute: _dropout,
    // %self.pre_rnn.lstm.training is a boolean value be used for attribute: train, bidirectional and batch_first.
    if (idx == 0 || idx == 1) {
        DLOG(INFO) << "prim::Constant attempts to set into the input of layer: " << layer_inID.first->getName();
        return false;
    } else if (idx == 2) {
        DLOG(INFO) << "weights of aten::lstm1 have been set in layer builder stage";
    } else if (idx == 3) {
        cur_layer->setHasBiases(getValueFromConstant<int>(d_tensor, layer_type, "has_biases"));
    } else if (idx == 4) {
        cur_layer->setNumLayers(getValueFromConstant<int64_t>(d_tensor, layer_type, "num_layers"));
    } else if (idx == 5) {
        cur_layer->setDropout(getValueFromConstant<double>(d_tensor, layer_type, "dropout"));
    } else if (idx == 6) {
        cur_layer->setTrain(getValueFromConstant<int>(d_tensor, layer_type, "train"));
    } else if (idx == 7) {
        cur_layer->setBidirectional(getValueFromConstant<int>(d_tensor, layer_type, "bidirectional"));
    } else if (idx == 8) {
        cur_layer->setBatchFirst(getValueFromConstant<int>(d_tensor, layer_type, "batch_first"));
    } else {
        DLOG(FATAL) << "Incorrect data from prim::Constant";
    }
    return true;
}

bool putAttributeInAtenLstm2(layer_inID_type& layer_inID, dtensor_ptr_type& d_tensor)
{
    auto cur_layer = std::dynamic_pointer_cast<ir::AtenLSTM2Layer>(layer_inID.first);
    auto idx = layer_inID.second;
    auto layer_type = cur_layer->getType();

    // e.g. %170 : Tensor, %171 : Tensor, %172 : Tensor = aten::lstm(%data.1, %batch_sizes.1, %164,
    //                                                               %self_model_encoder_rnn_layers_0__flat_weights.1,
    //                                                               %61, %149, %167, %51, %61)
    // %data.1, %batch_sizes.1, and %164 are inputs, %self_model_encoder_rnn_layers_0__flat_weights.1,
    // %61 is attribute: has_biases, %149 is attribute: _num_layers, %167 is attribute: _dropout,
    // %51, %61 is boolean values be used for attribute: train, bidirectional.
    if (idx == 0 || idx == 1 || idx == 2) {
        DLOG(INFO) << "prim::Constant attempts to set into the input of layer: " << layer_inID.first->getName();
        return false;
    } else if (idx == 3) {
        DLOG(INFO) << "weights of aten::lstm2 have been set in layer builder stage";
    } else if (idx == 4) {
        cur_layer->setHasBiases(getValueFromConstant<int>(d_tensor, layer_type, "has_biases"));
    } else if (idx == 5) {
        cur_layer->setNumLayers(getValueFromConstant<int64_t>(d_tensor, layer_type, "num_layers"));
    } else if (idx == 6) {
        cur_layer->setDropout(getValueFromConstant<double>(d_tensor, layer_type, "dropout"));
    } else if (idx == 7) {
        cur_layer->setTrain(getValueFromConstant<int>(d_tensor, layer_type, "train"));
    } else if (idx == 8) {
        cur_layer->setBidirectional(getValueFromConstant<int>(d_tensor, layer_type, "bidirectional"));
    } else {
        DLOG(FATAL) << "Incorrect data from prim::Constant";
    }
    return true;
}

bool putAttributeInAtenMax(layer_inID_type& layer_inID, dtensor_ptr_type& d_tensor)
{
    auto cur_layer = std::dynamic_pointer_cast<ir::AtenMaxLayer>(layer_inID.first);
    auto idx = layer_inID.second;
    auto layer_type = cur_layer->getType();

    // e.g. %v : Tensor, %k.1 : Tensor = aten::max(%logp.1, %83, %132)
    // %logp.1 is input, %83, %132 is attribute: dim, keep_dim.
    if (idx == 0) {
        DLOG(INFO) << "prim::Constant attempts to set into the input of layer: " << layer_inID.first->getName();
        return false;
    } else if (idx == 1) {
        cur_layer->setDim(getValueFromConstant<int>(d_tensor, layer_type, "dim"));
    } else if (idx == 2) {
        cur_layer->setKeepDim(getValueFromConstant<int>(d_tensor, layer_type, "keep_dim"));
    } else {
        DLOG(FATAL) << "Incorrect data from prim::Constant";
    }
    return true;
}

bool putAttributeInAtenMaxPool2d(layer_inID_type& layer_inID, dtensor_ptr_type& d_tensor)
{
    auto cur_layer = std::dynamic_pointer_cast<ir::AtenMaxPool2dLayer>(layer_inID.first);
    auto idx = layer_inID.second;
    auto layer_type = cur_layer->getType();

    // e.g.  Tensor = aten::max_pool2d(%x1.1, %61, %62, %64, %65, %66)
    // %x1.1 is input,
    // 961, %61, %62, %64, %65, %66 is attribute: kernel_size, pad, stride, dilation, ceil_mode
    if (idx == 0) {
        DLOG(INFO) << "prim::Constant attempts to set into the input of layer: " << layer_inID.first->getName();
        return false;
    } else if (idx == 1) {
        cur_layer->setKernelSize(getVectorFromConstant<int64_t>(d_tensor, layer_type, "kernel_size"));
    } else if (idx == 2) {
        cur_layer->setStride(getVectorFromConstant<int64_t>(d_tensor, layer_type, "stride"));
    } else if (idx == 3) {
        cur_layer->setPad(getVectorFromConstant<int64_t>(d_tensor, layer_type, "pad"));
    } else if (idx == 4) {
        cur_layer->setDilation(getVectorFromConstant<int64_t>(d_tensor, layer_type, "dilation"));
    } else if (idx == 5) {
        cur_layer->setCeilMode(getValueFromConstant<int>(d_tensor, layer_type, "ceil_mode"));
    } else {
        DLOG(FATAL) << "Incorrect data from prim::Constant";
    }
    return true;
}

bool putAttributeInAtenMin(layer_inID_type& layer_inID, dtensor_ptr_type& d_tensor)
{
    auto cur_layer = std::dynamic_pointer_cast<ir::AtenMinLayer>(layer_inID.first);
    auto idx = layer_inID.second;
    auto layer_type = cur_layer->getType();

    // e.g. Tensor, Tensor = aten::min(%eos_mask0.1, %149, %51)
    // %eos_mask0.1 is input, %149, %51 is attribute: dim_or_y, keep_dim.
    if (idx == 0) {
        DLOG(INFO) << "prim::Constant attempts to set into the input of layer: " << layer_inID.first->getName();
        return false;
    } else if (idx == 1) {
        cur_layer->setDimOrY(getValueFromConstant<int>(d_tensor, layer_type, "dim_or_y"));
    } else if (idx == 2) {
        cur_layer->setKeepDim(getValueFromConstant<int>(d_tensor, layer_type, "keep_dim"));
    } else {
        DLOG(FATAL) << "Incorrect data from prim::Constant";
    }
    return true;
}

bool putAttributeInAtenNorm(layer_inID_type& layer_inID, dtensor_ptr_type& d_tensor)
{
    auto cur_layer = std::dynamic_pointer_cast<ir::AtenNormLayer>(layer_inID.first);
    auto idx = layer_inID.second;
    auto layer_type = cur_layer->getType();

    // e.g. Tensor, Tensor = aten::norm(%1085, %101)
    // %1085 is input, %101 is attribute: p.
    if (idx == 0) {
        DLOG(INFO) << "prim::Constant attempts to set into the input of layer: " << layer_inID.first->getName();
        return false;
    } else if (idx == 1) {
        cur_layer->setP(getValueFromConstant<int64_t>(d_tensor, layer_type, "p"));
    } else {
        DLOG(FATAL) << "Incorrect data from prim::Constant";
    }
    return true;
}

bool putAttributeInAtenOnes(layer_inID_type& layer_inID, dtensor_ptr_type& d_tensor)
{
    auto cur_layer = std::dynamic_pointer_cast<ir::AtenOnesLayer>(layer_inID.first);
    auto idx = layer_inID.second;
    auto layer_type = cur_layer->getType();

    // e.g. Tensor = aten::ones(%521, %56, %57, %57, %57)
    // %521 is input, %56, %57, %57, %57 is attribute: dtype, layout, device, pin_memory.
    if (idx == 0) {
        DLOG(INFO) << "prim::Constant attempts to set into the input of layer: " << layer_inID.first->getName();
        return false;
    } else if (idx == 1) {
        auto data = d_tensor->getData<int64_t>();
        if ((*data).size()) {
            cur_layer->setDType((*data)[0]);
        } else {
            DLOG(INFO) << "dtype of aten::ones is set to NONE";
            return false;
        }
    } else if (idx == 2) {
        auto data = d_tensor->getData<int64_t>();
        if ((*data).size()) {
            cur_layer->setLayout((*data)[0]);
        } else {
            DLOG(INFO) << "layout of aten::ones is set to NONE";
            return false;
        }
    } else if (idx == 3) {
        cur_layer->setDevice(getStringFromConstant(d_tensor, cur_layer->getType(), "device"));
    } else if (idx == 4) {
        auto data = d_tensor->getData<int>();
        if ((*data).size()) {
            cur_layer->setPinMemory((*data)[0]);
        } else {
            DLOG(INFO) << "pin_memory of aten::ones is set to NONE";
            return false;
        }
    } else {
        DLOG(FATAL) << "Incorrect data from prim::Constant";
    }
    return true;
}

bool putAttributeInAtenPackPaddedSequence(layer_inID_type& layer_inID, dtensor_ptr_type& d_tensor)
{
    auto cur_layer = std::dynamic_pointer_cast<ir::AtenPackPaddedSequenceLayer>(layer_inID.first);
    auto idx = layer_inID.second;

    // e.g. Tensor = aten::_pack_padded_sequence(%x.1, %lengths.1, %61)
    // %x.1, %lengths.1 is input, %61 is attribute: batch_first.
    if (idx == 0 || idx == 1) {
        DLOG(INFO) << "prim::Constant attempts to set into the input of layer: " << layer_inID.first->getName();
        return false;
    } else if (idx == 2) {
        cur_layer->setBatchFirst(getValueFromConstant<int>(d_tensor, cur_layer->getType(), "batch_first"));
    } else {
        DLOG(FATAL) << "Incorrect data from prim::Constant";
    }
    return true;
}

bool putAttributeInAtenPadPackedSequence(layer_inID_type& layer_inID, dtensor_ptr_type& d_tensor)
{
    auto cur_layer = std::dynamic_pointer_cast<ir::AtenPadPackedSequenceLayer>(layer_inID.first);
    auto idx = layer_inID.second;
    auto layer_type = cur_layer->getType();

    // e.g. Tensor = aten::_pad_packed_sequence(%170, %batch_sizes.1, %61, %167, %max_seq_length.1)
    // %170 and %batch_sizes.1 are inputs,
    // %61 is attribute: batch_first.
    // %167 is attribute: padding_value.
    // %max_seq_length.1 is attribute: total_length.
    if (idx == 0 || idx == 1) {
        DLOG(INFO) << "prim::Constant attempts to set into the input of layer: " << layer_inID.first->getName();
        return false;
    } else if (idx == 2) {
        cur_layer->setBatchFirst(getValueFromConstant<int>(d_tensor, layer_type, "batch_first"));
    } else if (idx == 3) {
        cur_layer->setPaddingValue(getValueFromConstant<float>(d_tensor, layer_type, "padding_value"));
    } else if (idx == 4) {
        cur_layer->setTotalLength(getValueFromConstant<int64_t>(d_tensor, layer_type, "total_length"));
    } else {
        DLOG(FATAL) << "Incorrect data from prim::Constant";
    }
    return true;
}

bool putAttributeInAtenSelect(layer_inID_type& layer_inID, dtensor_ptr_type& d_tensor)
{
    auto cur_layer = std::dynamic_pointer_cast<ir::AtenSelectLayer>(layer_inID.first);
    auto idx = layer_inID.second;

    // e.g. Tensor = aten::select(%logits.1, %12, %batch_idx.1)
    // %logits.1 is input, %12 is attribute: dim, %batch_idx.1 is attribute: index.
    if (idx == 0) {
        DLOG(INFO) << "prim::Constant attempts to set into the input of layer: " << layer_inID.first->getName();
        return false;
    } else if (idx == 1) {
        cur_layer->setDim(getValueFromConstant<int64_t>(d_tensor, cur_layer->getType(), "dim"));
    } else if (idx == 2) {
        cur_layer->setIndex(getValueFromConstant<int64_t>(d_tensor, cur_layer->getType(), "index"));
    } else {
        DLOG(FATAL) << "Incorrect data from prim::Constant";
    }
    return true;
}

bool putAttributeInAtenSetItem(layer_inID_type& layer_inID, dtensor_ptr_type& d_tensor)
{
    auto cur_layer = std::dynamic_pointer_cast<ir::AtenSetItemLayer>(layer_inID.first);
    auto idx = layer_inID.second;

    // e.g. Tensor[] = aten::_set_item(%context_new.1, %counter.1, %576)
    // %context_new.1, %576 is input,
    // %counter.1 is attribute: indices.
    if (idx == 0 || idx == 2) {
        DLOG(INFO) << "prim::Constant attempts to set into the input of layer: " << layer_inID.first->getName();
        return false;
    } else if (idx == 1) {
        cur_layer->setIndices(getValueFromConstant<int>(d_tensor, cur_layer->getType(), "indices"));
    } else {
        DLOG(FATAL) << "Incorrect data from prim::Constant";
    }
    return true;
}

bool putAttributeInAtenSize(layer_inID_type& layer_inID, dtensor_ptr_type& d_tensor)
{
    auto cur_layer = std::dynamic_pointer_cast<ir::AtenSizeLayer>(layer_inID.first);
    auto idx = layer_inID.second;

    // e.g. int = aten::size(%x_padded.1, %5)
    // %x_padded.1 is input, %5 is attribute: dim.
    if (idx == 0) {
        DLOG(INFO) << "prim::Constant attempts to set into the input of layer: " << layer_inID.first->getName();
        return false;
    } else if (idx == 1) {
        cur_layer->setDim(getValueFromConstant<int64_t>(d_tensor, cur_layer->getType(), "dim"));
    } else {
        DLOG(FATAL) << "Incorrect data from prim::Constant";
    }
    return true;
}

bool putAttributeInAtenSlice(layer_inID_type& layer_inID, dtensor_ptr_type& d_tensor)
{
    auto cur_layer = std::dynamic_pointer_cast<ir::AtenSliceLayer>(layer_inID.first);
    auto idx = layer_inID.second;
    auto layer_type = cur_layer->getType();

    // e.g. Tensor = aten::slice(%22, %12, %12, %26, %33)
    // %22 is input, %12 is attribute: dim, %12 is attribute: start, %26 is attribute: end, %33 is attribute: step
    if (idx == 0) {
        DLOG(INFO) << "prim::Constant attempts to set into the input of layer: " << layer_inID.first->getName();
        return false;
    } else if (idx == 1) {
        cur_layer->setDim(getValueFromConstant<int64_t>(d_tensor, layer_type, "dim"));
    } else if (idx == 2) {
        cur_layer->setStart(getValueFromConstant<int64_t>(d_tensor, layer_type, "start"));
    } else if (idx == 3) {
        cur_layer->setEnd(getValueFromConstant<int64_t>(d_tensor, layer_type, "end"));
    } else if (idx == 4) {
        cur_layer->setStep(getValueFromConstant<int64_t>(d_tensor, layer_type, "step"));
    } else {
        DLOG(FATAL) << "Incorrect data from prim::Constant";
    }
    return true;
}

bool putAttributeInAtenSoftmax(layer_inID_type& layer_inID, dtensor_ptr_type& d_tensor)
{
    auto cur_layer = std::dynamic_pointer_cast<ir::AtenSoftmaxLayer>(layer_inID.first);
    auto idx = layer_inID.second;
    auto layer_type = cur_layer->getType();

    // e.g. Tensor = aten::softmax(%scores0.1, %117, %57)
    // %scores0.1 is input,%117, %57 is attribute: dim, dtype.
    if (idx == 0) {
        DLOG(INFO) << "prim::Constant attempts to set into the input of layer: " << layer_inID.first->getName();
        return false;
    } else if (idx == 1) {
        cur_layer->setDim(getValueFromConstant<int64_t>(d_tensor, layer_type, "dim"));
    } else if (idx == 2) {
        auto data = d_tensor->getData<int64_t>();
        if ((*data).size()) {
            cur_layer->setDtype((*data)[0]);
        } else {
            DLOG(INFO) << "dtype of aten::softmax is set to NONE";
            return false;
        }
    } else {
        DLOG(FATAL) << "Incorrect data from prim::Constant";
    }
    return true;
}

bool putAttributeInAtenSqueeze(layer_inID_type& layer_inID, dtensor_ptr_type& d_tensor)
{
    auto cur_layer = std::dynamic_pointer_cast<ir::AtenSqueezeLayer>(layer_inID.first);
    auto idx = layer_inID.second;

    // e.g. Tensor = aten::squeeze(%1447, %149)
    // %1447 is input, %149 is attribute: dim,
    if (idx == 0) {
        DLOG(INFO) << "prim::Constant attempts to set into the input of layer: " << layer_inID.first->getName();
        return false;
    } else if (idx == 1) {
        cur_layer->setDim(getValueFromConstant<int64_t>(d_tensor, cur_layer->getType(), "dim"));
    } else {
        DLOG(FATAL) << "Incorrect data from prim::Constant";
    }
    return true;
}

bool putAttributeInAtenSub(layer_inID_type& layer_inID, dtensor_ptr_type& d_tensor)
{
    auto cur_layer = std::dynamic_pointer_cast<ir::AtenSubLayer>(layer_inID.first);
    auto idx = layer_inID.second;

    // e.g.
    // (1) %symbols_subed0.1 : int = aten::sub(%symbols_subed.3, %39)
    //     %symbols_subed.3 and %39 are inputs;
    // (2) 480 : Tensor,  = aten::sub (473, 443, 59, )
    //     473 and 443 are inputs, 59 is attribute: alpha.
    if (idx == 0 || idx == 1) {
        DLOG(INFO) << "prim::Constant attempts to set into the input of layer: " << layer_inID.first->getName();
        return false;
    } else if (idx == 2) {
        cur_layer->setAlpha(getValueFromConstant<int64_t>(d_tensor, cur_layer->getType(), "alpha"));
    } else {
        DLOG(FATAL) << "Incorrect data from prim::Constant";
    }
    return true;
}

bool putAttributeInAtenSum(layer_inID_type& layer_inID, dtensor_ptr_type& d_tensor)
{
    auto cur_layer = std::dynamic_pointer_cast<ir::AtenSumLayer>(layer_inID.first);
    auto idx = layer_inID.second;
    auto layer_type = cur_layer->getType();

    // e.g. Tensor = aten::sum(%penalty1.1, %1714, %51, %57)
    // %penalty1.1 is input, %1714, %51, %57 is attribute: dim, keepdim, dtype.
    if (idx == 0) {
        DLOG(INFO) << "prim::Constant attempts to set into the input of layer: " << layer_inID.first->getName();
        return false;
    } else if (idx == 1) {
        cur_layer->setDim(getVectorFromConstant<int64_t>(d_tensor, layer_type, "dim"));
    } else if (idx == 2) {
        cur_layer->setKeepdim(getValueFromConstant<int>(d_tensor, layer_type, "keepdim"));
    } else if (idx == 3) {
        auto data = d_tensor->getData<int64_t>();
        if ((*data).size()) {
            cur_layer->setDtype((*data)[0]);
        } else {
            DLOG(INFO) << "dtype of aten::sum is set to NONE";
            return false;
        }
    } else {
        DLOG(FATAL) << "Incorrect data from prim::Constant";
    }
    return true;
}

bool putAttributeInAtenTo1(layer_inID_type& layer_inID, dtensor_ptr_type& d_tensor)
{
    auto cur_layer = std::dynamic_pointer_cast<ir::AtenTo1Layer>(layer_inID.first);
    auto idx = layer_inID.second;
    auto layer_type = cur_layer->getType();

    // e.g.
    // Tensor = aten::to(%x_lens.1, %99, %self.pre_rnn.lstm.training, %self.pre_rnn.lstm.training, %3)
    // %x_lens.1 is input, %99 is attribute: dtype,
    // %self.pre_rnn.lstm.training is assigned to attribute: non_blocking and copy both,
    // %3 is attribute: optional_memory_format.
    if (idx == 0) {
        DLOG(INFO) << "prim::Constant attempts to set into the input of layer: " << layer_inID.first->getName();
        return false;
    } else if (idx == 1) {
        cur_layer->setDType(getValueFromConstant<int64_t>(d_tensor, layer_type, "dtype"));
    } else if (idx == 2) {
        cur_layer->setNonBlocking(getValueFromConstant<int>(d_tensor, layer_type, "non_blocking"));
    } else if (idx == 3) {
        cur_layer->setCopy(getValueFromConstant<int>(d_tensor, layer_type, "copy"));
    } else if (idx == 4) {
        // optional_memory_format could be NONE
        auto data = d_tensor->getData<int64_t>();
        if ((*data).size()) {
            cur_layer->setOptionalMemoryFormat((*data)[0]);
        } else {
            DLOG(INFO) << "NONE is set to attribute optional_memory_format of aten::to";
        }
    } else {
        DLOG(FATAL) << "Incorrect data from prim::Constant";
    }
    return true;
}

bool putAttributeInAtenTo2(layer_inID_type& layer_inID, dtensor_ptr_type& d_tensor)
{
    auto cur_layer = std::dynamic_pointer_cast<ir::AtenTo2Layer>(layer_inID.first);
    auto idx = layer_inID.second;
    auto layer_type = cur_layer->getType();

    // e.g.
    // Tensor = aten::to(%linear_att.1, %1090, %56, %56, %62)
    // %linear_att.1, %1090 are inputs.
    // %56 is assigned to attribute: non_blocking and copy both,
    // %62 is attribute: optional_memory_format.
    if (idx == 0 || idx == 1) {
        DLOG(INFO) << "prim::Constant attempts to set into the input of layer: " << layer_inID.first->getName();
        return false;
    } else if (idx == 2) {
        cur_layer->setNonBlocking(getValueFromConstant<int>(d_tensor, layer_type, "non_blocking"));
    } else if (idx == 3) {
        cur_layer->setCopy(getValueFromConstant<int>(d_tensor, layer_type, "copy"));
    } else if (idx == 4) {
        // optional_memory_format could be NONE
        auto data = d_tensor->getData<int64_t>();
        if ((*data).size()) {
            cur_layer->setOptionalMemoryFormat((*data)[0]);
        } else {
            DLOG(INFO) << "NONE is set to attribute optional_memory_format of aten::to";
        }
    } else {
        DLOG(FATAL) << "Incorrect data from prim::Constant";
    }
    return true;
}

bool putAttributeInAtenTopk(layer_inID_type& layer_inID, dtensor_ptr_type& d_tensor)
{
    auto cur_layer = std::dynamic_pointer_cast<ir::AtenTopkLayer>(layer_inID.first);
    auto idx = layer_inID.second;
    auto layer_type = cur_layer->getType();

    // e.g. Tensor, Tensor = aten::topk(%ret1.1, %511, %117, %61, %61)
    // %ret1.1 is input, %511, %117, %61, %61 is attribute: k, dim, largest, sorted
    if (idx == 0) {
        DLOG(INFO) << "prim::Constant attempts to set into the input of layer: " << layer_inID.first->getName();
        return false;
    } else if (idx == 1) {
        cur_layer->setK(getValueFromConstant<int>(d_tensor, layer_type, "K"));
    } else if (idx == 2) {
        cur_layer->setDim(getValueFromConstant<int>(d_tensor, layer_type, "dim"));
    } else if (idx == 3) {
        cur_layer->setLargest(getValueFromConstant<int>(d_tensor, layer_type, "largest"));
    } else if (idx == 4) {
        cur_layer->setSorted(getValueFromConstant<int>(d_tensor, layer_type, "sorted"));
    } else {
        DLOG(FATAL) << "Incorrect data from prim::Constant";
    }
    return true;
}

bool putAttributeInAtenTranspose(layer_inID_type& layer_inID, dtensor_ptr_type& d_tensor)
{
    auto cur_layer = std::dynamic_pointer_cast<ir::AtenTransposeLayer>(layer_inID.first);
    auto idx = layer_inID.second;

    // e.g. Tensor = aten::transpose(%x_padded.7, %4, %5)
    // %x_padded.7 is input, $ is attribute: dim0, %5 is attribute: dim1.
    if (idx == 0) {
        DLOG(INFO) << "prim::Constant attempts to set into the input of layer: " << layer_inID.first->getName();
        return false;
    } else if (idx == 1) {
        cur_layer->setDim0(getValueFromConstant<int64_t>(d_tensor, cur_layer->getType(), "dim0"));
    } else if (idx == 2) {
        cur_layer->setDim1(getValueFromConstant<int64_t>(d_tensor, cur_layer->getType(), "dim1"));
    } else {
        DLOG(FATAL) << "Incorrect data from prim::Constant";
    }
    return true;
}

bool putAttributeInAtenUnsqueeze(layer_inID_type& layer_inID, dtensor_ptr_type& d_tensor)
{
    auto cur_layer = std::dynamic_pointer_cast<ir::AtenUnsqueezeLayer>(layer_inID.first);
    auto idx = layer_inID.second;

    // e.g. Tensor = aten::unsqueeze(%32, %33)
    // %32 is input, %33 is attribute: dim.
    if (idx == 0) {
        DLOG(INFO) << "prim::Constant attempts to set into the input of layer: " << layer_inID.first->getName();
        return false;
    } else if (idx == 1) {
        cur_layer->setDim(getValueFromConstant<int64_t>(d_tensor, cur_layer->getType(), "dim"));
    } else {
        DLOG(FATAL) << "Incorrect data from prim::Constant";
    }
    return true;
}

bool putAttributeInAtenWarn(layer_inID_type& layer_inID, dtensor_ptr_type& d_tensor)
{
    auto cur_layer = std::dynamic_pointer_cast<ir::AtenWarnLayer>(layer_inID.first);
    auto idx = layer_inID.second;

    // e.g. = aten::warn[warn_id=0](%233, %210)
    // %233 is input, %210 is attribute: value.
    if (idx == 0) {
        DLOG(INFO) << "prim::Constant attempts to set into the input of layer: " << layer_inID.first->getName();
        return false;
    } else if (idx == 1) {
        cur_layer->setValue(getValueFromConstant<int>(d_tensor, cur_layer->getType(), "value"));
    } else {
        DLOG(FATAL) << "Incorrect data from prim::Constant";
    }
    return true;
}

bool putAttributeInPrimLoop(layer_inID_type& layer_inID, dtensor_ptr_type& d_tensor)
{
    auto cur_layer = std::dynamic_pointer_cast<ir::PrimLoopLayer>(layer_inID.first);
    auto idx = layer_inID.second;

    // e.g. = prim::Loop(%13, %16, inputs)
    // %13 is attribute: trip_count, %16 is attribute: cond.
    if (idx == 0) {
        cur_layer->setTripCount(getValueFromConstant<int64_t>(d_tensor, cur_layer->getType(), "trip_count"));
    } else if (idx == 1) {
        cur_layer->setCond(getValueFromConstant<int64_t>(d_tensor, cur_layer->getType(), "cond"));
    } else {
        DLOG(INFO) << "prim::Constant attempts to set into the input of layer: " << layer_inID.first->getName();
        return false;
    }
    return true;
}

bool putAttributeInPrimTupleIndex(layer_inID_type& layer_inID, dtensor_ptr_type& d_tensor)
{
    auto cur_layer = std::dynamic_pointer_cast<ir::PrimTupleIndexLayer>(layer_inID.first);
    auto idx = layer_inID.second;

    // e.g. Tensor = prim::TupleIndex(%hx, %7)
    // %hx is input, %7 is attribute: index.
    if (idx == 0) {
        DLOG(INFO) << "prim::Constant attempts to set into the input of layer: " << layer_inID.first->getName();
        return false;
    } else if (idx == 1) {
        cur_layer->setIndex(getValueFromConstant<int64_t>(d_tensor, cur_layer->getType(), "index"));
    } else {
        DLOG(FATAL) << "Incorrect data from prim::Constant";
    }
    return true;
}

std::string getStringFromConstant(dtensor_ptr_type& d_tensor, nn_compiler::ir::LayerType layer_type,
                                  std::string attr_type)
{
    std::string str = "";
    auto data = d_tensor->getData<uint8_t>();
    if ((*data).size()) {
        for (unsigned int i = 0; i < (*data).size(); i++) {
            str += static_cast<char>((*data)[i]);
        }
    } else {
        DLOG(FATAL) << attr_type << " of " << convertLayerTypeToString(layer_type) << " cannot be NONE";
    }
    return str;
}

AttributeHelper::AttributeHelper()
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
    type_to_function_["aten::clone"] = &putAttributeInAtenClone;
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
    type_to_function_["aten::layer_norm"] = &putAttributeInAtenLayerNorm;
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

bool AttributeHelper::putAttribute(std::string name, layer_inID_type& layer_inID, dtensor_ptr_type& d_tensor)
{
    bool ret_code = false;
    std::map<std::string, putAttributeFunc>::iterator iter = type_to_function_.find(name);
    if (iter == type_to_function_.end()) {
        DLOG(INFO) << convertLayerTypeToString(layer_inID.first->getType())
                   << " don't require attribute setting, or this prim::Constant is used as input.";
    } else {
        ret_code = (*(iter->second))(layer_inID, d_tensor);
    }

    return ret_code;
}

}  // namespace optimizer_utils
}  // namespace frontend
}  // namespace nn_compiler
