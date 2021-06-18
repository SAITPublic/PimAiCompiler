DEPTHWISE_BY_DEFAULT_NODE(nn_ir::NodeType::ELTWISE, nn_ir::EltwiseNode)
DEPTHWISE_BY_DEFAULT_NODE(nn_ir::NodeType::MAAELTWISE, nn_ir::MAAEltwiseNode)
DEPTHWISE_BY_DEFAULT_NODE(nn_ir::NodeType::POOL, nn_ir::PoolNode)
DEPTHWISE_BY_DEFAULT_NODE(nn_ir::NodeType::DATAFORMAT, nn_ir::DataFormatNode)
DEPTHWISE_BY_DEFAULT_NODE(nn_ir::NodeType::DUMMY, nn_ir::DummyNode)
#undef DEPTHWISE_BY_DEFAULT_NODE
