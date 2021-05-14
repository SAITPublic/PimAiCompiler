RASTER_IFM_NODE(nn_ir::PermuteNode, !N->isUsingCellInput())
RASTER_IFM_NODE(nn_ir::ReshapeNode, true)
RASTER_IFM_NODE(nn_ir::QuantNode, true)
RASTER_IFM_NODE(nn_ir::DequantNode, true)
RASTER_IFM_NODE(nn_ir::DataFormatNode, N->getFormatDirection() == DataFormatConversion::TENSOR2CELL)
// FIXME: Temporary for testing. In fact CopyNode inherits this
// property fron surrounding nodes, just like MemoryHint
RASTER_IFM_NODE(nn_ir::CopyNode, N->isDramToDram())
#undef RASTER_IFM_NODE
