RASTER_OFM_NODE(nn_ir::PermuteNode, true)
RASTER_OFM_NODE(nn_ir::PriorBoxNode, true)
RASTER_OFM_NODE(nn_ir::ReshapeNode, true)
RASTER_OFM_NODE(nn_ir::QuantNode, true)
RASTER_OFM_NODE(nn_ir::DequantNode, true)
RASTER_OFM_NODE(nn_ir::DataFormatNode, N->getFormatDirection() == DataFormatConversion::CELL2TENSOR)
// FIXME: Temporary for testing. In fact CopyNode inherits this
// property fron surrounding nodes, just like MemoryHint
RASTER_OFM_NODE(nn_ir::CopyNode, N->isDramToDram())
#undef RASTER_OFM_NODE
