#include "builder/model_builder.h"
#include "compiler/include/nn_compiler.hpp"

namespace nncir = nn_compiler::nn_ir;

namespace nnrt
{
RetVal ModelBuilder::compileModel(const int compile_level, const std::string model_type)
{
    nn_compiler::NNCompiler compiler;
    std::vector<std::shared_ptr<nncir::NNIR>> NNIR_graphs;
    compiler.initialize(compile_level, this->model_path_, model_type);
    compiler.compile(NNIR_graphs);
    compiler.finalize();
    this->runnable_ir_ = NNIR_graphs.front();

    return RetVal::SUCCESS;
}

RetVal ModelBuilder::preloadModel()
{
    for (auto& op_node : runnable_ir_->getNodes()) {
        if (op_node.getNodeType() == nncir::NodeType::ATENLSTM1 ||
                   op_node.getNodeType() == nncir::NodeType::ATENLSTM2 ||
                   op_node.getNodeType() == nncir::NodeType::ATENCONV2D ||
                   op_node.getNodeType() == nncir::NodeType::ATENBATCHNORM2D ||
                   op_node.getNodeType() == nncir::NodeType::ATENLINEAR) {
            // For Ops' with weight/bias, preload weights/bias to blobs_container.
            std::vector<nncir::Blob*> weight_blobs;
            std::vector<nncir::Blob*> bias_blobs;

            if (op_node.getNodeType() == nncir::NodeType::ATENLSTM1) {
                auto lstm_node = cast<nncir::AtenLSTM1Node>(op_node);
                weight_blobs = lstm_node.getWeightBlob();
                bias_blobs = lstm_node.getBiasBlob();
            } else if (op_node.getNodeType() == nncir::NodeType::ATENLSTM2) {
                auto lstm_node = cast<nncir::AtenLSTM2Node>(op_node);
                weight_blobs = lstm_node.getWeightBlob();
                bias_blobs = lstm_node.getBiasBlob();
            } else if (op_node.getNodeType() == nncir::NodeType::ATENCONV2D) {
                auto conv2d_node = cast<nncir::AtenConv2dNode>(op_node);
                weight_blobs = conv2d_node.getWeightBlob();
                bias_blobs = conv2d_node.getBiasBlob();
            } else if (op_node.getNodeType() == nncir::NodeType::ATENBATCHNORM2D) {
                auto bn2d_node = cast<nncir::AtenBatchNorm2dNode>(op_node);
                weight_blobs = bn2d_node.getWeightBlob();
                bias_blobs = bn2d_node.getBiasBlob();
            } else if (op_node.getNodeType() == nncir::NodeType::ATENLINEAR) {
                auto linear_node = cast<nncir::AtenLinearNode>(op_node);
                weight_blobs = linear_node.getWeightBlob();
                bias_blobs = linear_node.getBiasBlob();
            }

            for (auto blob : weight_blobs) {
                this->loadWeightAndBias(blob);
            }
            for (auto blob : bias_blobs) {
                this->loadWeightAndBias(blob);
            }
        }
    }

    return RetVal::SUCCESS;
}

std::pair<std::shared_ptr<nncir::NNIR>, ModelBuilder::blob_store_type> ModelBuilder::getModel() {
    return std::make_pair(this->runnable_ir_, this->preloaded_blobs_container_);
}

void ModelBuilder::loadWeightAndBias(nncir::Blob* blob)
{
    nncir::Shape4D shape = blob->getShape();
    int64_t blob_id = blob->getId();

    auto data_blob = cast_if<nncir::DataBlob>(blob);
    if (data_blob == nullptr) {
        LOG(FATAL) << "The blob is not weight or bias blob!";
    }

    auto bit_width = blob->getBitWidth();
    at::ScalarType scalar_type;
    // torch::Tensor tensor_data;

    std::vector<int64_t> shape_arr;
    if (shape.n > 0) shape_arr.push_back(shape.n);
    if (shape.c > 0) shape_arr.push_back(shape.c);
    if (shape.h > 0) shape_arr.push_back(shape.h);
    if (shape.w > 0) shape_arr.push_back(shape.w);

    if (bit_width == 16) {
        auto value_vec = data_blob->getBuf<float16>();
        scalar_type = torch::kHalf;
        auto tensor_data = at::from_blob(value_vec.data(), shape_arr, scalar_type).cuda();
        torch::jit::IValue iv = torch::jit::IValue(tensor_data);
        this->preloaded_blobs_container_.insert({blob_id, {DataType::TENSOR, iv}});
    } else if (bit_width == 32) {
        auto value_vec = data_blob->getBuf<float>();
        scalar_type = torch::kFloat;
        auto tensor_data = at::from_blob(value_vec.data(), shape_arr, scalar_type).cuda();
        torch::jit::IValue iv = torch::jit::IValue(tensor_data);
        this->preloaded_blobs_container_.insert({blob_id, {DataType::TENSOR, iv}});
    } else {
        LOG(FATAL) << "Bit witdh Error!";
    }
}

}  // namespace nnrt
