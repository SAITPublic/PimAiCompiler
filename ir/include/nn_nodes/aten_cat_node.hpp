#pragma once

#include "common/include/common.hpp"
#include "common/include/types.hpp"

#include "ir/include/ir_types.hpp"
#include "ir/include/nn_node.hpp"

namespace nn_compiler {
namespace nn_ir {

class AtenCatNode : public NodeMixin<AtenCatNode, NNNode> {
 public:
    explicit AtenCatNode(const NodeInfo& node_info, int64_t dim, int64_t mem_blob_id = -1)
        : NodeMixin(node_info, NodeType::ATENCAT), dim_(dim), mem_blob_id_(mem_blob_id) {}

    std::string getNodeTypeAsString() const override { return "AtenCat"; }

    int64_t getDim() const { return dim_; }

    int64_t getMemBlobId() const { return mem_blob_id_; }

    void setMemBlobId(int64_t mem_blob_id) { mem_blob_id_ = mem_blob_id; }

 private:
    int64_t dim_;
    int64_t mem_blob_id_;

}; // class AtenCatNode

} // namespace nn_ir
} // namespace nn_compiler
