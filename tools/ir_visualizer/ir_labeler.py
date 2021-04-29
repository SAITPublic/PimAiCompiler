import numpy as np

import IR.Blob
import IR.NNNode.ActivationNode
import IR.NNNode.AnyType
import IR.NNNode.BatchNormNode
import IR.NNNode.ConcatNode
import IR.NNNode.ConvNode
import IR.NNNode.CopyNode
import IR.NNNode.DeConvNode
import IR.NNNode.EltwiseNode
import IR.NNNode.FullyConnectedNode
import IR.NNNode.InputNode
import IR.NNNode.PoolNode
import IR.NNNode.ScaleNode
import IR.NNNode.PermuteNode
import IR.NNNode.ReshapeNode
import IR.NnNode
import IR.Node
import IR.Type.TypedValue
import IR.OpNode
import IR.OPNode.AnyType
import IR.OPNode.AddNode
import IR.NNNode.MatMulNode
import IR.OPNode.ShiftNode
import ir_enums
import IR.NNNode.DataFormatNode
import IR.NNNode.SliceNode
import IR.NNNode.TileNode


def node_name(graph_id: int, node_id: int) -> str:
    return str(graph_id) + '_node' + str(node_id)


def blob_node_name(graph_id: int, blob_id: int) -> str:
    return str(graph_id) + '_blob' + str(blob_id)


def edge_name(graph_id: int, edge_id: int) -> str:
    return str(graph_id) + '_edge' + str(edge_id)


def head_node_name(graph_id: int) -> str:
    return str(graph_id) + '_head'


def tail_node_name(graph_id: int) -> str:
    return str(graph_id) + '_tail'


# Labels for NNNode -> AnyType
def nn_node_label(nn_node: IR.NnNode.NnNode) -> (object, str):
    node_type = nn_node.NnNodeType()
    if node_type == IR.NNNode.AnyType.AnyType().InputNode:
        nn_node_input_node = IR.NNNode.InputNode.InputNode()
        nn_node_input_node.Init(nn_node.NnNode().Bytes, nn_node.NnNode().Pos)
        return nn_node_input_node, input_node_label(nn_node_input_node)

    elif node_type == IR.NNNode.AnyType.AnyType().ConvNode:
        nn_node_conv_node = IR.NNNode.ConvNode.ConvNode()
        nn_node_conv_node.Init(nn_node.NnNode().Bytes, nn_node.NnNode().Pos)
        return nn_node_conv_node, conv_node_label(nn_node_conv_node)

    elif node_type == IR.NNNode.AnyType.AnyType().DeConvNode:
        nn_node_deconv_node = IR.NNNode.DeConvNode.DeConvNode()
        nn_node_deconv_node.Init(nn_node.NnNode().Bytes, nn_node.NnNode().Pos)
        return nn_node_deconv_node, deconv_node_label(nn_node_deconv_node)

    elif node_type == IR.NNNode.AnyType.AnyType().PoolNode:
        nn_node_pool_node = IR.NNNode.PoolNode.PoolNode()
        nn_node_pool_node.Init(nn_node.NnNode().Bytes, nn_node.NnNode().Pos)
        return nn_node_pool_node, pool_node_label(nn_node_pool_node)

    elif node_type == IR.NNNode.AnyType.AnyType().ActivationNode:
        nn_node_activation_node = IR.NNNode.ActivationNode.ActivationNode()
        nn_node_activation_node.Init(nn_node.NnNode().Bytes, nn_node.NnNode().Pos)
        return nn_node_activation_node, activation_node_label(nn_node_activation_node)

    elif node_type == IR.NNNode.AnyType.AnyType().FullyConnectedNode:
        nn_node_fc_node = IR.NNNode.FullyConnectedNode.FullyConnectedNode()
        nn_node_fc_node.Init(nn_node.NnNode().Bytes, nn_node.NnNode().Pos)
        return nn_node_fc_node, fully_connected_node_label(nn_node_fc_node)

    elif node_type == IR.NNNode.AnyType.AnyType().ConcatNode:
        nn_node_cc_node = IR.NNNode.ConcatNode.ConcatNode()
        nn_node_cc_node.Init(nn_node.NnNode().Bytes, nn_node.NnNode().Pos)
        return nn_node_cc_node, concat_node_label(nn_node_cc_node)

    elif node_type == IR.NNNode.AnyType.AnyType().EltwiseNode:
        nn_node_ew_node = IR.NNNode.EltwiseNode.EltwiseNode()
        nn_node_ew_node.Init(nn_node.NnNode().Bytes, nn_node.NnNode().Pos)
        return nn_node_ew_node, eltwise_node_label(nn_node_ew_node)

    elif node_type == IR.NNNode.AnyType.AnyType().BatchNormNode:
        nn_node_bn_node = IR.NNNode.BatchNormNode.BatchNormNode()
        nn_node_bn_node.Init(nn_node.NnNode().Bytes, nn_node.NnNode().Pos)
        return nn_node_bn_node, batch_norm_node_label(nn_node_bn_node)

    elif node_type == IR.NNNode.AnyType.AnyType().ScaleNode:
        nn_node_scale_node = IR.NNNode.ScaleNode.ScaleNode()
        nn_node_scale_node.Init(nn_node.NnNode().Bytes, nn_node.NnNode().Pos)
        return nn_node_scale_node, scale_node_label(nn_node_scale_node)

    elif node_type == IR.NNNode.AnyType.AnyType().PermuteNode:
        nn_node_permute_node = IR.NNNode.PermuteNode.PermuteNode()
        nn_node_permute_node.Init(nn_node.NnNode().Bytes, nn_node.NnNode().Pos)
        return nn_node_permute_node, permute_node_label(nn_node_permute_node)

    elif node_type == IR.NNNode.AnyType.AnyType().ReshapeNode:
        nn_node_reshape_node = IR.NNNode.ReshapeNode.ReshapeNode()
        nn_node_reshape_node.Init(nn_node.NnNode().Bytes, nn_node.NnNode().Pos)
        return nn_node_reshape_node, reshape_node_label(nn_node_reshape_node)

    elif node_type == IR.NNNode.AnyType.AnyType().SliceNode:
        slice_node = IR.NNNode.SliceNode.SliceNode()
        slice_node.Init(nn_node.NnNode().Bytes, nn_node.NnNode().Pos)
        return slice_node, slice_node_label(slice_node)

    elif node_type == IR.NNNode.AnyType.AnyType().TileNode:
        tile_node = IR.NNNode.TileNode.TileNode()
        tile_node.Init(nn_node.NnNode().Bytes, nn_node.NnNode().Pos)
        return tile_node, tile_node_label(tile_node)

    elif node_type == IR.NNNode.AnyType.AnyType().MatMulNode:
        mm_node = IR.NNNode.MatMulNode.MatMulNode()
        mm_node.Init(nn_node.NnNode().Bytes, nn_node.NnNode().Pos)
        return mm_node, list_table(['MatMul Node'])

    elif node_type == IR.NNNode.AnyType.AnyType().CopyNode:
        copy_node = IR.NNNode.CopyNode.CopyNode()
        copy_node.Init(nn_node.NnNode().Bytes, nn_node.NnNode().Pos)
        return copy_node, list_table(['Copy Node'])

    else:
        return None, list_table(['Unknown NNNode Type'])


def op_node_label(op_node: IR.OpNode.OpNode) -> (object, str):
    node_type = op_node.OpNodeType()
    if node_type == IR.OPNode.AnyType.AnyType().AddNode:
        op_node_add_node = IR.OPNode.AddNode.AddNode()
        op_node_add_node.Init(op_node.OpNode().Bytes, op_node.OpNode().Pos)
        return op_node_add_node, list_table(['Add Node'])
    elif node_type == IR.OPNode.AnyType.AnyType().ShiftNode:
        op_node_shift_node = IR.OPNode.ShiftNode.ShiftNode()
        op_node_shift_node.Init(op_node.OpNode().Bytes, op_node.OpNode().Pos)
        return op_node_shift_node, shift_node_label(op_node_shift_node)
    else:
        return None, list_table(['Unknown OpNode Type'])


def input_node_label(input_node: IR.NNNode.InputNode.InputNode) -> str:
    retval = 'Input Node ({})<br/>'.format(ir_enums.InputType.get_name(input_node.Type()))
    if type(input_node.MeanAsNumpy()) is np.ndarray:
        retval += 'mean: {}<br/>'.format(input_node.MeanAsNumpy())
    retval += 'scale:{}<br/>'.format(input_node.Scale())
    retval += 'mirror: {}'.format(input_node.Mirror())

    return list_table([retval])


def conv_node_label(conv_node: IR.NNNode.ConvNode.ConvNode) -> str:
    kernel_size = conv_node.KernelSize()
    stride_size = conv_node.StrideSize()
    dilation_size = conv_node.DilationSize()
    padding_size = conv_node.PaddingSize()
    activation_node = conv_node.Activation()
    shift_node = conv_node.Shift()

    retval = 'Conv Node<br/>'
    if kernel_size is not None:
        retval += 'Kernel Size: {}<br/>'.format(dim2_label(kernel_size))
    if stride_size is not None:
        retval += 'Stride Size: {}<br/>'.format(dim2_label(stride_size))
    if dilation_size is not None:
        retval += 'Dilation Size: {}<br/>'.format(dim2_label(dilation_size))
    if padding_size is not None:
        retval += 'Padding Size: {}<br/>'.format(pad4_label(padding_size))

    activation_label = None
    if activation_node is not None:
        activation_label = activation_node_label(activation_node)
    shift_label = None
    if shift_node is not None:
        shift_label = shift_node_label(shift_node)

    return list_table([retval, activation_label, shift_label])


def deconv_node_label(deconv_node: IR.NNNode.DeConvNode.DeConvNode) -> str:
    kernel_size = deconv_node.KernelSize()
    stride_size = deconv_node.StrideSize()
    dilation_size = deconv_node.DilationSize()
    padding_size = deconv_node.PaddingSize()
    sampling_size = deconv_node.SamplingSize()
    output_padding = deconv_node.OutputPadding()
    activation_node = deconv_node.Activation()
    shift_node = deconv_node.Shift()

    retval = 'DeConv Node<br/>'
    if kernel_size is not None:
        retval += 'Kernel Size: {}<br/>'.format(dim2_label(kernel_size))
    if stride_size is not None:
        retval += 'Stride Size: {}<br/>'.format(dim2_label(stride_size))
    if dilation_size is not None:
        retval += 'Dilation Size: {}<br/>'.format(dim2_label(dilation_size))
    if padding_size is not None:
        retval += 'Padding Size: {}<br/>'.format(pad4_label(padding_size))

    if sampling_size is not None:
        retval += 'Sampling Size: {}<br/>'.format(dim2_label(sampling_size))
    retval += 'Interpolation: {}<br/>'.format(deconv_node.Interpolation())
    if output_padding is not None:
        retval += 'Output Padding: {}<br/>'.format(dim2_label(output_padding))

    activation_label = None
    if activation_node is not None:
        activation_label = activation_node_label(activation_node)

    shift_label = None
    if shift_node is not None:
        shift_label = shift_node_label(shift_node)

    return list_table([retval, activation_label, shift_label])


def pool_node_label(pool_node: IR.NNNode.PoolNode.PoolNode) -> str:
    kernel_size = pool_node.KernelSize()
    stride_size = pool_node.StrideSize()
    dilation_size = pool_node.DilationSize()
    padding_size = pool_node.PaddingSize()

    retval = 'Pool Node ({})<br/>'.format(ir_enums.PoolType.get_name(pool_node.Type()))
    if kernel_size is not None:
        retval += 'KernelSize: {}<br/>'.format(dim2_label(kernel_size))
    if stride_size is not None:
        retval += 'StrideSize: {}<br/>'.format(dim2_label(stride_size))
    if dilation_size is not None:
        retval += 'DilationSize: {}<br/>'.format(dim2_label(dilation_size))
    if padding_size is not None:
        retval += 'PaddingSize: {}<br/>'.format(pad4_label(padding_size))
    retval += 'PadCalculation: {}<br/>'.format(ir_enums.PadCalculation.get_name(pool_node.PadCalc()))
    return list_table([retval])


def activation_node_label(activation_node: IR.NNNode.ActivationNode.ActivationNode) -> str:
    retval = 'Activation Node ({})<br/>'.format(ir_enums.ActivationType.get_name(activation_node.Type()))
    retval += 'Slope: {:.2f}<br/>'.format(activation_node.Slope())
    retval += 'NegativeSlope: {:.2f}<br/>'.format(activation_node.NegativeSlope())
    retval += 'Min: {:.2f}, Max: {:.2f}<br/>'.format(activation_node.Min(), activation_node.Max())
    return list_table([retval])


def fully_connected_node_label(fc_node: IR.NNNode.FullyConnectedNode.FullyConnectedNode) -> str:
    activation_node = fc_node.Activation()
    shift_node = fc_node.Shift()

    retval = 'Fully Connected Node<br/>'
    retval += 'Axis: {}<br/>'.format(fc_node.Axis())
    retval += 'Transpose: {}<br/>'.format(fc_node.Transpose())

    activation_label = None
    if activation_node is not None:
        activation_label = activation_node_label(activation_node)

    shift_label = None
    if shift_node is not None:
        shift_label = shift_node_label(shift_node)

    return list_table([retval, activation_label, shift_label])


def concat_node_label(concat_node: IR.NNNode.ConcatNode.ConcatNode) -> str:
    retval = 'Concat Node<br/>'
    retval += 'Axis: {}'.format(concat_node.Axis())
    return list_table([retval])


def eltwise_node_label(eltwise_node: IR.NNNode.EltwiseNode.EltwiseNode) -> str:
    retval = 'Eltwise Node<br/>'
    retval += 'Operation: {}<br/>'.format(ir_enums.EltwiseType.get_name(eltwise_node.Operation()))
    retval += 'Stable Prod Grad: {}'.format(eltwise_node.StableProdGrad())
    return list_table([retval])


def batch_norm_node_label(batch_norm_node: IR.NNNode.BatchNormNode.BatchNormNode) -> str:
    retval = 'BatchNorm Node<br/>'
    retval += 'Axis: {}<br/>'.format(batch_norm_node.Axis())
    retval += 'Use Global Stats: {}<br/>'.format(batch_norm_node.UseGlobalStats())
    retval += 'Scale: {}<br/>'.format(batch_norm_node.Scale())
    retval += 'Eps: {}<br/>'.format(batch_norm_node.Eps())
    return list_table([retval])


def scale_node_label(scale_node: IR.NNNode.ScaleNode.ScaleNode) -> str:
    retval = 'Scale Node<br/>'
    retval += 'BiasTerm: {}<br/>'.format(scale_node.BiasTerm())
    return list_table([retval])


def permute_node_label(permute_node: IR.NNNode.PermuteNode.PermuteNode) -> str:
    retval = 'Permute Node<br/>'
    retval += 'Order: {}<br/>'.format(dim4_label(permute_node.Order()))
    return list_table([retval])


def reshape_node_label(reshape_node: IR.NNNode.ReshapeNode.ReshapeNode) -> str:
    retval = 'ReshapeNode Node<br/>'
    retval += 'Shape: {}<br/>'.format(dim4_label(reshape_node.Shape()))
    return list_table([retval])


def slice_node_label(slice_node: IR.NNNode.SliceNode.SliceNode) -> str:
    retval = 'Slice Node<br/>'
    retval += 'Axis: {}<br/>'.format(slice_node.Axis())
    retval += 'Points: {}<br/>'.format(typed_array_label(slice_node.Points()))
    return list_table([retval])


def tile_node_label(tile_node: IR.NNNode.TileNode.TileNode) -> str:
    retval = 'Tile Node<br/>'
    retval += 'Axis: {}<br/>'.format(tile_node.Axis())
    retval += 'Tiles: {}<br/>'.format(tile_node.Tiles())
    return list_table([retval])


# Labels for OpNode
def shift_node_label(shift_node: IR.OPNode.ShiftNode.ShiftNode) -> str:
    quantization_shift = shift_node.QuantizationShiftAsNumpy()
    multiplication_shift = shift_node.MultiplicationShiftAsNumpy()
    activation_shift = shift_node.ActivationShiftAsNumpy()
    lut_scale = shift_node.LutScaleAsNumpy()
    lut_bias = shift_node.LutBiasAsNumpy()
    grelu_info = shift_node.GreluInfoAsNumpy()

    retval = 'Shift Node<br/>'
    if type(quantization_shift) is np.ndarray and quantization_shift.any():
        retval += 'Quantization Shift: {}<br/>'.format(quantization_shift.shape)
    if type(multiplication_shift) is np.ndarray and multiplication_shift.any():
        retval += 'Multiplication Shift: {}<br/>'.format(multiplication_shift)
    if type(activation_shift) is np.ndarray and activation_shift.any():
        retval += 'Activation Shift: {}<br/>'.format(activation_shift)
    if type(lut_scale) is np.ndarray and lut_scale.any():
        retval += 'Lut Scale: {}<br/>'.format(lut_scale)
    if type(lut_bias) is np.ndarray and lut_bias.any():
        retval += 'Lut Bias: {}<br/>'.format(lut_bias)
    if type(grelu_info) is np.ndarray and grelu_info.any():
        retval += 'Grelu Info: {}<br/>'.format(grelu_info)

    return list_table([retval])


# Labels for Blob
def data_edge_label(ir_data_edge: ir_enums.EdgeType.DataEdge) -> str:
    return list_table(['Data Edge Blob'])


def kernel_blob_edge_label(conv_node: IR.NNNode.ConvNode.ConvNode) -> str:
    return list_table(['Kernel Blob'])


def bias_blob_edge_label(conv_node: IR.NNNode.ConvNode.ConvNode) -> str:
    return list_table(['Bias Blob'])


def weight_blob_edge_label(fc_node: IR.NNNode.FullyConnectedNode.FullyConnectedNode) -> str:
    return list_table(['Weight Blob'])


def blob_node_label(blob: IR.Blob.Blob) -> str:
    retval = ''
    retval += 'QuantType: {}<br/>'.format(ir_enums.QuantType.get_name(blob.QuantType()))
    retval += 'ShapeType: {}<br/>'.format(ir_enums.ShapeType.get_name(blob.Shape()))
    retval += 'Dim: {}<br/>'.format(dim4_label(blob.Dim()))
    retval += 'DataType: {}<br/>'.format(ir_enums.DataType.get_name(blob.DataType()))
    retval += 'BitWidth: {}<br/>'.format(blob.BitWidth())
    if blob.HwInfo():
        hwinfo = blob.HwInfo()
        retval += 'LivenessNode: {} - {}<br/>'.format(hwinfo.LivenessStartNode(),
                                                      hwinfo.LivenessEndNode())
        retval += 'Alignment: {}<br/>'.format(dim4_label(hwinfo.AlignmentUnit()))
    retval += 'ZeroPoint: {}<br/>'.format(blob.ZeroPoint())
    # FIXME: frac_len and quant_leval should be handled later
    frac_len = []
    if blob.FracLenLength() <= 5:
        for idx in range(0, blob.FracLenLength()):
            frac_len.append(blob.FracLen(idx))
    else:
        for idx in range(0, 3):
            frac_len.append(blob.FracLen(idx))
        frac_len.append('...')
        frac_len.append(blob.FracLen(blob.FracLenLength() - 1))
    retval += 'FracLen: {} size: {}<br/>'.format(frac_len, blob.FracLenLength())
    return list_table([blob_title(blob), retval])


def blob_title(blob: IR.Blob.Blob) -> str:
    return bold('[BLOB:{}]<br/>{}'.format(blob.Id(), blob.Name().decode('utf-8')))


def ir_node_title(ir_node: IR.Node.Node) -> str:
    return bold('[NODE:{}]<br/>{}'.format(ir_node.Id(), ir_node.Name().decode('utf-8')))


def ir_edge_title(ir_edge: IR.Edge.Edge, hide_name: bool) -> str:
    title = '[EDGE:{}]'.format(ir_edge.Id())
    if not hide_name:
        title += '<br/>{}'.format(ir_edge.Name().decode('utf-8'))
    return bold(title)


# for types
def dim2_label(dim2: IR.Type.Dim2.Dim2) -> str:
    return '(h:{} w:{})'.format(dim2.H(), dim2.W())


def dim4_label(dim4: IR.Type.Dim4.Dim4) -> str:
    if dim4 is None:
        return '(No dim)'
    return '(n:{} c:{} h:{} w:{})'.format(dim4.N(), dim4.C(), dim4.H(), dim4.W())


def pad4_label(pad4: IR.Type.Pad4.Pad4) -> str:
    return '(t:{} b:{} l:{} r:{})'.format(pad4.T(), pad4.B(), pad4.L(), pad4.R())


def typed_array_label(typed_array: IR.Type.TypedArray.TypedArray) -> str:
    as_f32_arr = typed_array.F32ArrAsNumpy()
    as_i16_arr = typed_array.I16ArrAsNumpy()
    as_i8_arr = typed_array.I8ArrAsNumpy()
    as_ui8_arr = typed_array.Ui8ArrAsNumpy()
    as_i64_arr = typed_array.I64ArrAsNumpy()

    if type(as_f32_arr) is np.ndarray:
        return '{}'.format(as_f32_arr)
    if type(as_i16_arr) is np.ndarray:
        return '{}'.format(as_i16_arr)
    if type(as_i8_arr) is np.ndarray:
        return '{}'.format(as_i8_arr)
    if type(as_ui8_arr) is np.ndarray:
        return '{}'.format(as_ui8_arr)
    if type(as_i64_arr) is np.ndarray:
        return '{}'.format(as_i64_arr)
    return 'Unknown type in TypedArray'


# for offsets and sizes
def offset_or_size_label(value: int) -> str:
    ''' Returns the integer value provided as formatted string with "decimal (hex)" format '''
    return '{} (0x{:x})'.format(value, value)


# for html elements
def html_str(text: str) -> str:
    return '<{}>'.format(text)


def table(tr) -> str:
    return '<table BORDER="0" cellspacing="0">{}</table>'.format(tr)


def list_table(rows: list) -> str:
    rows = [x for x in rows if x is not None]
    for num, row in enumerate(rows):
        if row is None:
            continue
        if str(row).startswith('<tr>') and str(row).endswith('</tr>'):
            rows[num] = str(row)
        elif '<hr/>' == str(row):
            rows[num] = '<tr><td></td></tr><hr/><tr><td></td></tr>'
        else:
            rows[num] = tr(str(row))
    return table(''.join(rows))


def tr(text: str, align='center') -> str:
    return '<tr><td align="{0}" balign="{0}">{1}</td></tr>'.format(align, text)


def bold(text: str) -> str:
    return '<b>{}</b>'.format(text)
