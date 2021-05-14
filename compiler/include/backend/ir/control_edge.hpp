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
 * @file.    control_edge.hpp
 * @brief.   This is ControlEdge class
 * @details. This header defines ControlEdge class.
 * @version. 0.1.
 */

#pragma once

#include "common/common.hpp"
#include "common/types.hpp"

#include "ir/edge.hpp"
#include "ir/ir_types.hpp"

namespace nn_compiler {
namespace nn_ir {

class ControlEdge : public Edge {
 public:
    // to be update

    template <typename T>
    static bool classof(const Edge* edge) {
        static_assert(std::is_same<T, ControlEdge>::value, "incorrect type");
        return edge->getEdgeType() == EdgeType::CONTROL;
    }

 private:
}; // class ControlEdge

} // namespace nn_ir
} // namespace nn_compiler
