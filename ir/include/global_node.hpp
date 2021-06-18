/*
 * Copyright (C) 2019 Samsung Electronics Co. LTD
 *
 * This software is proprietary of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

/**
 * @file.    global_node.hpp
 * @brief.   This is GlobalNode class
 * @details. This header defines GlobalNode class.
 * @version. 0.1.
 */

#pragma once

#include "common/include/common.hpp"
#include "common/include/types.hpp"

#include "ir/include/ir_types.hpp"
#include "ir/include/node.hpp"

namespace nn_compiler {
namespace nn_ir {

class GlobalNode : public AbstractNodeMixin<GlobalNode, Node> {
 public:
    /**
     * @brief.      Constructor of GlobalNode.
     * @details.    This function constructs GlobalNode
     * @param[in].
     * @param[out].
     * @returns.
     */

    virtual Shape4D    getOriginalDim() const = 0;
    virtual MemoryInfo getMemoryInfo() const  = 0;
    NODE_ID_T          getUId() const { return uid_; }
    SyncType           getSyncType() const { return sync_type_; }
    void               removeSync() { sync_type_ = SyncType::NONE; }
    void               setSync() { sync_type_ = SyncType::LOCAL; }
    SigType            getSigType() const { return sig_type_; }

 protected:
    GlobalNode(const GlobalNode&) = default;
    // constructor for inherited classes
    explicit GlobalNode(const NodeInfo& node_info,
                        NodeType        node_type,
                        NODE_ID_T       uid,
                        nn_ir::SyncType sync_type,
                        nn_ir::SigType  sig_type)
        : AbstractNodeMixin(node_info, node_type), uid_(uid), sync_type_(sync_type), sig_type_(sig_type) {}

 private:
    NODE_ID_T uid_;
    SyncType  sync_type_;
    SigType   sig_type_;
}; // class GlobalNode

} // namespace nn_ir
} // namespace nn_compiler
