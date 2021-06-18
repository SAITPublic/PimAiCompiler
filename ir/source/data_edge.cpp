/*
 * Copyright (C) 2019 Samsung Electronics Co. LTD
 *
 * This software is proprietary of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#include "ir/data_edge.hpp"
#include "ir/nn_ir.hpp"

namespace nn_compiler {
namespace nn_ir {

Blob* DataEdge::getBlob() const { return blob_id_ == INVALID_ID ? nullptr : getGraph().getBlob(blob_id_); }

} // namespace nn_ir
} // namespace nn_compiler
