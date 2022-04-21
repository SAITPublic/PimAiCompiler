#include "ir/include/types.h"

namespace nn_compiler
{
namespace ir
{
std::string convertLayerTypeToString(LayerType type)
{
    static std::unordered_map<LayerType, std::string> converter;
    converter.insert({LayerType::ATENADD, "aten::add"});
    converter.insert({LayerType::ATENADDMM, "aten::addmm"});
    converter.insert({LayerType::ATENAND, "aten::__and__"});
    converter.insert({LayerType::ATENANY, "aten::any"});
    converter.insert({LayerType::ATENAPPEND, "aten::append"});
    converter.insert({LayerType::ATENARANGE1, "aten::arange1"});
    converter.insert({LayerType::ATENARANGE2, "aten::arange2"});
    converter.insert({LayerType::ATENARANGE3, "aten::arange3"});
    converter.insert({LayerType::ATENASTENSOR, "aten::as_tensor"});
    converter.insert({LayerType::ATENBATCHNORM2D, "aten::batch_norm"});
    converter.insert({LayerType::ATENBITWISENOT, "aten::bitwise_not"});
    converter.insert({LayerType::ATENBMM, "aten::bmm"});
    converter.insert({LayerType::ATENBOOL, "aten::Bool"});
    converter.insert({LayerType::ATENCAT, "aten::cat"});
    converter.insert({LayerType::ATENCEIL, "aten::ceil"});
    converter.insert({LayerType::ATENCHUNK, "aten::chunk"});
    converter.insert({LayerType::ATENCLAMP, "aten::clamp"});
    converter.insert({LayerType::ATENCLEAR, "aten::clear"});
    converter.insert({LayerType::ATENCLONE, "aten::clone"});
    converter.insert({LayerType::ATENCONTIGUOUS, "aten::contiguous"});
    converter.insert({LayerType::ATENCONV2D, "aten::conv2d"});
    converter.insert({LayerType::ATENCOPY, "aten::copy_"});
    converter.insert({LayerType::ATENCPU, "aten::cpu"});
    converter.insert({LayerType::ATENCUDA, "aten::cuda"});
    converter.insert({LayerType::ATENDERIVEINDEX, "aten::__derive_index"});
    converter.insert({LayerType::ATENDETACH, "aten::detach"});
    converter.insert({LayerType::ATENDIM, "aten::dim"});
    converter.insert({LayerType::ATENDIV, "aten::div"});
    converter.insert({LayerType::ATENDROPOUT, "aten::dropout"});
    converter.insert({LayerType::ATENEMBEDDING, "aten::embedding"});
    converter.insert({LayerType::ATENEQ, "aten::eq"});
    converter.insert({LayerType::ATENEQUAL, "aten::equal"});
    converter.insert({LayerType::ATENEXPAND, "aten::expand"});
    converter.insert({LayerType::ATENFILL, "aten::fill_"});
    converter.insert({LayerType::ATENFLOORDIVIDE, "aten::floor_divide"});
    converter.insert({LayerType::ATENFORMAT, "aten::format"});
    converter.insert({LayerType::ATENGETITEM, "aten::__getitem__"});
    converter.insert({LayerType::ATENGATHER, "aten::gather"});
    converter.insert({LayerType::ATENGE, "aten::ge"});
    converter.insert({LayerType::ATENGT, "aten::gt"});
    converter.insert({LayerType::ATENINDEX, "aten::index"});
    converter.insert({LayerType::ATENINDEXPUT, "aten::index_put_"});
    converter.insert({LayerType::ATENINDEXSELECT, "aten::index_select"});
    converter.insert({LayerType::ATENINT, "aten::Int"});
    converter.insert({LayerType::ATENINTIMPLICIT, "aten::IntImplicit"});
    converter.insert({LayerType::ATENITEM, "aten::item"});
    converter.insert({LayerType::ATENIS, "aten::__is__"});
    converter.insert({LayerType::ATENLAYERNORM, "aten::layer_norm"});
    converter.insert({LayerType::ATENLEAKYRELU, "aten::leaky_relu"});
    converter.insert({LayerType::ATENLEN, "aten::len"});
    converter.insert({LayerType::ATENLINEAR, "aten::linear"});
    converter.insert({LayerType::ATENLIST, "aten::list"});
    converter.insert({LayerType::ATENLOG, "aten::log"});
    converter.insert({LayerType::ATENLOGSOFTMAX, "aten::log_softmax"});
    converter.insert({LayerType::ATENLSTM1, "aten::lstm1"});
    converter.insert({LayerType::ATENLSTM2, "aten::lstm2"});
    converter.insert({LayerType::ATENLT, "aten::lt"});
    converter.insert({LayerType::ATENMASKEDFILL, "aten::masked_fill"});
    converter.insert({LayerType::ATENMASKEDSELECT, "aten::masked_select"});
    converter.insert({LayerType::ATENMATMUL, "aten::matmul"});
    converter.insert({LayerType::ATENMAX, "aten::max"});
    converter.insert({LayerType::ATENMAXPOOL2D, "aten::max_pool2d"});
    converter.insert({LayerType::ATENMIN, "aten::min"});
    converter.insert({LayerType::ATENMUL, "aten::mul"});
    converter.insert({LayerType::ATENNE, "aten::ne"});
    converter.insert({LayerType::ATENNEG, "aten::neg"});
    converter.insert({LayerType::ATENNORM, "aten::norm"});
    converter.insert({LayerType::ATENNOT, "aten::not"});
    converter.insert({LayerType::ATENONES, "aten::ones"});
    converter.insert({LayerType::ATENPACKPADDEDSEQUENCE, "aten::_pack_padded_sequence"});
    converter.insert({LayerType::ATENPADPACKEDSEQUENCE, "aten::_pad_packed_sequence"});
    converter.insert({LayerType::ATENPOW, "aten::pow"});
    converter.insert({LayerType::ATENRELU, "aten::relu"});
    converter.insert({LayerType::ATENRESHAPE, "aten::reshape"});
    converter.insert({LayerType::ATENREMAINDER, "aten::remainder"});
    converter.insert({LayerType::ATENREPEAT, "aten::repeat"});
    converter.insert({LayerType::ATENSELECT, "aten::select"});
    converter.insert({LayerType::ATENSETITEM, "aten::_set_item"});
    converter.insert({LayerType::ATENSIZE, "aten::size"});
    converter.insert({LayerType::ATENSLICE, "aten::slice"});
    converter.insert({LayerType::ATENSOFTMAX, "aten::softmax"});
    converter.insert({LayerType::ATENSQUEEZE, "aten::squeeze"});
    converter.insert({LayerType::ATENSUB, "aten::sub"});
    converter.insert({LayerType::ATENSUM, "aten::sum"});
    converter.insert({LayerType::ATENTANH, "aten::tanh"});
    converter.insert({LayerType::ATENTENSOR, "aten::tensor"});
    converter.insert({LayerType::ATENTRANSPOSE, "aten::transpose"});
    converter.insert({LayerType::ATENTO1, "aten::to1"});
    converter.insert({LayerType::ATENTO2, "aten::to2"});
    converter.insert({LayerType::ATENTOPK, "aten::topk"});
    converter.insert({LayerType::ATENTRIU, "aten::triu"});
    converter.insert({LayerType::ATENUNSQUEEZE, "aten::unsqueeze"});
    converter.insert({LayerType::ATENVIEW, "aten::view"});
    converter.insert({LayerType::ATENWARN, "aten::warn"});
    converter.insert({LayerType::ATENZEROS, "aten::zeros"});
    converter.insert({LayerType::ATENZEROSLIKE, "aten::zeros_like"});

    converter.insert({LayerType::PRIMBLOCK, "prim::Block"});
    converter.insert({LayerType::PRIMCALLMETHOD, "prim::CallMethod"});
    converter.insert({LayerType::PRIMCONSTANT, "prim::Constant"});
    converter.insert({LayerType::PRIMDATA, "prim::data"});
    converter.insert({LayerType::PRIMDEVICE, "prim::device"});
    converter.insert({LayerType::PRIMDTYPE, "prim::dtype"});
    converter.insert({LayerType::PRIMENDIF, "prim::EndIf"});
    converter.insert({LayerType::PRIMENDLOOP, "prim::EndLoop"});
    converter.insert({LayerType::PRIMGETATTR, "prim::GetAttr"});
    converter.insert({LayerType::PRIMIF, "prim::If"});
    converter.insert({LayerType::PRIMINPUT, "prim::Input"});
    converter.insert({LayerType::PRIMLISTCONSTRUCT, "prim::ListConstruct"});
    converter.insert({LayerType::PRIMLISTUNPACK, "prim::ListUnpack"});
    converter.insert({LayerType::PRIMLOOP, "prim::Loop"});
    converter.insert({LayerType::PRIMLOOPINDEX, "prim::LoopIndex"});
    converter.insert({LayerType::PRIMOUTPUT, "prim::Output"});
    converter.insert({LayerType::PRIMRAISEEXCEPTION, "prim::RaiseException"});
    converter.insert({LayerType::PRIMSETATTR, "prim::SetAttr"});
    converter.insert({LayerType::PRIMTOLIST, "prim::tolist"});
    converter.insert({LayerType::PRIMTUPLECONSTRUCT, "prim::TupleConstruct"});
    converter.insert({LayerType::PRIMTUPLEINDEX, "prim::TupleIndex"});
    converter.insert({LayerType::PRIMTUPLEUNPACK, "prim::TupleUnpack"});
    converter.insert({LayerType::PRIMTYPE, "prim::type"});
    converter.insert({LayerType::PRIMUNCHECKEDCAST, "prim::unchecked_cast"});
    converter.insert({LayerType::PRIMUNINITIALIZED, "prim::Uninitialized"});
    converter.insert({LayerType::PRIMVARIABLE, "prim::Variable"});

    auto iter = converter.find(type);
    if (iter == converter.end()) {
        DLOG(FATAL) << "Found unsupported layer type.";
    } else {
        return iter->second;
    }
}

}  // namespace ir
}  // namespace nn_compiler
