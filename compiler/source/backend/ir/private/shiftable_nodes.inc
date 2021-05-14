SHIFTABLE_NODE(nn_ir::NodeType::CONVOLUTION, nn_ir::ConvolutionNode)
SHIFTABLE_NODE(nn_ir::NodeType::DECONVOLUTION, nn_ir::DeconvolutionNode)
SHIFTABLE_NODE(nn_ir::NodeType::ELTWISE, nn_ir::EltwiseNode)
SHIFTABLE_NODE(nn_ir::NodeType::MAAELTWISE, nn_ir::MAAEltwiseNode)
SHIFTABLE_NODE(nn_ir::NodeType::ACTIVATION, nn_ir::ActivationNode)
SHIFTABLE_NODE(nn_ir::NodeType::FULLYCONNECTED, nn_ir::FullyConnectedNode)
SHIFTABLE_NODE(nn_ir::NodeType::MATMUL, nn_ir::MatMulNode)
#undef SHIFTABLE_NODE
