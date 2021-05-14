ACTIVATABLE_NODE(nn_ir::NodeType::CONVOLUTION, nn_ir::ConvolutionNode)
ACTIVATABLE_NODE(nn_ir::NodeType::DECONVOLUTION, nn_ir::DeconvolutionNode)
ACTIVATABLE_NODE(nn_ir::NodeType::MAAELTWISE, nn_ir::MAAEltwiseNode)
ACTIVATABLE_NODE(nn_ir::NodeType::FULLYCONNECTED, nn_ir::FullyConnectedNode)
#undef ACTIVATABLE_NODE
