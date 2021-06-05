import datetime
import argparse
import os
import sys

import numpy as np
from graphviz import Digraph

import IR.AnyNode
import IR.Blob
import IR.Edge
import IR.Graph
import IR.NnNode
import IR.Node
import IR.OpNode
import IR.Root
import IR.Type.DataType
import IR.Type.Dim2
import IR.Type.Dim4
import IR.Type.Pad4
import IR.Type.QuantType
import IR.Type.ShapeType
import IR.Type.TypedArray
import IR.Type.TypedValue
import IR.Type.EdgeType

import IR.NNNode.AtenAddNode
import IR.NNNode.AtenAppendNode
import IR.NNNode.AtenCatNode
import IR.NNNode.AtenCeilNode
import IR.NNNode.AtenCopyNode
import IR.NNNode.AtenDeriveIndexNode
import IR.NNNode.AtenDimNode
import IR.NNNode.AtenDivNode
import IR.NNNode.AtenDropoutNode
import IR.NNNode.AtenEmbeddingNode
import IR.NNNode.AtenEqNode
import IR.NNNode.AtenExpandNode
import IR.NNNode.AtenFormatNode
import IR.NNNode.AtenGetItemNode
import IR.NNNode.AtenGtNode
import IR.NNNode.AtenIntNode
import IR.NNNode.AtenIsNode
import IR.NNNode.AtenItemNode
import IR.NNNode.AtenLSTMNode
import IR.NNNode.AtenLenNode
import IR.NNNode.AtenListNode
import IR.NNNode.AtenNeNode
import IR.NNNode.AtenNegNode
import IR.NNNode.AtenSelectNode
import IR.NNNode.AtenSizeNode
import IR.NNNode.AtenSliceNode
import IR.NNNode.AtenTensorNode
import IR.NNNode.AtenToNode
import IR.NNNode.AtenUnsqueezeNode
import IR.NNNode.AtenZerosLikeNode
import IR.NNNode.AtenZerosNode

import IR.ControlNode
import ir_labeler as labeler

from distutils.util import strtobool


class Visualizer():

    def __init__(self, input_file: str, args: str):
        # do config set
        self.__set_config(args)
        # read flatbuffer binary and Create Root Object
        self.input_file = input_file
        self.ir_root = IR.Root.Root.GetRootAsRoot(self.read_flattbuffer_bin(input_file), 0)

        self.p_graph = Digraph(format=self.__file_format,
                               engine=self.__engine,
                               node_attr={'fontsize': self.__node_config['font_size'],
                                          'fontcolor': self.__node_config['font_color']},
                               edge_attr={'fontsize': self.__edge_blob_config['font_size']},
                               graph_attr={'rankdir': self.__rank_dir})
        self.__named_ir_graphs = {}
        self.__named_viz_graphs = {}
        self.__main_ir_graph_name = None
        self.__main_viz_graph_name = None
    
    def show_ir_graph_info(self):
        total_nodes = 0
        total_blobs = 0
        total_edges = 0
        graphs_length = self.ir_root.GraphsLength()
        for graph_num in range(graphs_length):
            ir_graph = self.ir_root.Graphs(graph_num)
            graph_name = ir_graph.Name().decode('utf-8')
            num_nodes = ir_graph.NodesLength()
            num_blobs = ir_graph.BlobsLength()
            num_edges = ir_graph.EdgesLength()
            total_nodes += num_nodes
            total_blobs += num_blobs
            total_edges += num_edges
            print('Idx:{} GraphName:{} NumOfNodes:{} NumOfBlobs:{} NumOfEdges:{}'.format(graph_num, graph_name, num_nodes, num_blobs, num_edges))
        print('\nSummary')
        print('*'*50)
        print('Total Graphs:{}'.format(graphs_length))
        print('Total Nodes(Ops):{}'.format(total_nodes))
        print('Total Blobs:{}'.format(total_blobs))
        print('Total Edges:{}'.format(total_edges))
        print('*'*50)
        print()

    def __set_config(self, args: str):

        self.__engine = str(args.engine)
        self.__file_format = str(args.format)
        self.__rank_dir = str(args.rank_dir)
        self.__hide_blobs = strtobool(args.hide_blobs)
        print("hide_edge_names = ", args.hide_edge_names)
        self.__hide_edge_names = strtobool(args.hide_edge_names)
        self.__hide_instr = strtobool(args.hide_instr)
        self.__hide_meminfos = strtobool(args.hide_meminfos)

        self.__node_config = dict()
        self.__node_config['font_size'] = str(args.node_font_size)
        self.__node_config['font_color'] = str(args.node_font_color)

        self.__edge_config = dict()
        self.__edge_config['font_size'] = str(args.edge_font_size)
        self.__edge_config['font_color'] = str(args.edge_font_color)
        self.__edge_config['color'] = str(args.edge_color)

        self.__kernel_blob_config = dict()
        self.__kernel_blob_config['font_size'] = str(args.kernel_blob_font_size)
        self.__kernel_blob_config['font_color'] = str(args.kernel_blob_font_color)
        self.__kernel_blob_config['color'] = str(args.kernel_blob_color)

        self.__bias_blob_config = dict()
        self.__bias_blob_config['font_size'] = str(args.bias_blob_font_size)
        self.__bias_blob_config['font_color'] = str(args.bias_blob_font_color)
        self.__bias_blob_config['color'] = str(args.bias_blob_color)

        self.__edge_blob_config = dict()
        self.__edge_blob_config['font_size'] = str(args.edge_blob_font_size)
        self.__edge_blob_config['font_color'] = str(args.edge_blob_font_color)
        self.__edge_blob_config['color'] = str(args.edge_blob_color)

        self.__lut_blob_config = dict()
        self.__lut_blob_config['font_size'] = str(args.lut_blob_font_size)
        self.__lut_blob_config['font_color'] = str(args.lut_blob_font_color)
        self.__lut_blob_config['color'] = str(args.lut_blob_color)

    def read_flattbuffer_bin(self, filename: str) -> bytes:
        if os.path.exists(filename):
            with open(filename, 'rb') as binfile:
                try:
                    return binfile.read()
                except IOError:
                    print('Can Not Open {}'.format(filename))
                    exit()
        else:
            print('{} Does Not Exist'.format(filename))
            exit()
    
    def draw(self):
        for graph_num in range(self.ir_root.GraphsLength()):
            ir_graph = self.ir_root.Graphs(graph_num)
            nw_cluster = Digraph(name='cluster_{}_nw'.format(graph_num),
                                graph_attr={'label': ir_graph.Name().decode('utf-8')})
            self.__node_mem_infos = []

            self.__blob_dict = dict()
            # node iteration
            self.__draw_nodes(ir_graph=ir_graph, cluster=nw_cluster)

            # edge iteration
            self.__draw_edges(ir_graph=ir_graph, cluster=nw_cluster)

            # blob iteration
            if not self.__hide_blobs:
                self.__draw_blobs(ir_graph=ir_graph, cluster=nw_cluster)

            self.p_graph.subgraph(nw_cluster)
            # break


    def __draw_nodes(self, ir_graph: IR.Graph.Graph, cluster: Digraph):
        for node_num in range(ir_graph.NodesLength()):
            ir_node = ir_graph.Nodes(node_num)
            ir_node_title = labeler.ir_node_title(ir_node)
            node_name = labeler.node_name(ir_graph.Id(), ir_node.Id())
            # handle IR.Node by Type
            if ir_node.NodeType() == IR.AnyNode.AnyNode().NnNode:
                ir_nn_node = IR.NnNode.NnNode()
                ir_nn_node.Init(ir_node.Node().Bytes, ir_node.Node().Pos)

                # Create Object for each NNNode.Type
                typed_node, nn_node_label = labeler.nn_node_label(ir_nn_node)
                node_label = labeler.list_table([ir_node_title, nn_node_label])

 
                # Save edge's blob info
                if 'WeightBlobIds' in dir(typed_node):
                    # LSTM have >= 8 Weight, it is a list
                    weight_blob_id_list = [typed_node.WeightBlobIds(k) for k in range(typed_node.WeightBlobIdsLength())]
                    edge_label = labeler.html_str(labeler.aten_weight_blob_edge_label())
                    if weight_blob_id_list:
                        for blob_id in weight_blob_id_list:
                            self.__append_blob_dict(blob_id, blob_type='weight', dst=node_name, label=edge_label)
                if 'BiasBlobIds' in dir(typed_node):
                    bias_blob_id_list = bias_blob_id_list = [typed_node.BiasBlobIds(k) for k in range(typed_node.BiasBlobIdsLength())]
                    edge_label = labeler.html_str(labeler.aten_bias_blob_edge_label())
                    if bias_blob_id_list:
                        for blob_id in bias_blob_id_list:
                            self.__append_blob_dict(blob_id, blob_type='bias', dst=node_name, label=edge_label)

                if 'KernelBlobId' in dir(typed_node):
                    blob_id = typed_node.KernelBlobId()
                    edge_label = labeler.html_str(labeler.kernel_blob_edge_label(typed_node))
                    self.__append_blob_dict(blob_id, blob_type='kernel', dst=node_name, label=edge_label)

                if 'BiasBlobId' in dir(typed_node):
                    blob_id = typed_node.BiasBlobId()
                    edge_label = labeler.html_str(labeler.bias_blob_edge_label(typed_node))
                    self.__append_blob_dict(blob_id, blob_type='bias', dst=node_name, label=edge_label)

                if 'ExpLutBlobId' in dir(typed_node):
                    blob_id = typed_node.ExpLutBlobId()
                    self.__append_blob_dict(blob_id, blob_type='lut', dst=node_name, label='')

                if 'SoftmaxLutBlobId' in dir(typed_node):
                    blob_id = typed_node.SoftmaxLutBlobId()
                    self.__append_blob_dict(blob_id, blob_type='lut', dst=node_name, label='')

            elif ir_node.NodeType() == IR.AnyNode.AnyNode().OpNode:
                ir_op_node = IR.OpNode.OpNode()
                ir_op_node.Init(ir_node.Node().Bytes, ir_node.Node().Pos)

                # Create Object for each OPNode.Type
                typed_node, op_node_label = labeler.op_node_label(ir_op_node)
                node_label = labeler.list_table([ir_node_title, op_node_label])
            
            # torch::prim Ops
            elif ir_node.NodeType() == IR.AnyNode.AnyNode().ControlNode:
                # create prim node
                ir_control_node = IR.ControlNode.ControlNode()
                ir_control_node.Init(ir_node.Node().Bytes, ir_node.Node().Pos)

                typed_node, control_node_label = labeler.control_node_label(ir_control_node)
                node_label = labeler.list_table([ir_node_title, control_node_label])

            else:
                # Handle Unknown Node
                node_label = labeler.list_table([ir_node_title, 'Unknown Node'])

            # Draw
            cluster.node(node_name, labeler.html_str(node_label), shape='rectangle')


    def __draw_edges(self, ir_graph: IR.Graph.Graph, cluster: Digraph):
        # set attr
        cluster.attr('node', fontsize=self.__edge_config['font_size'])
        cluster.attr('node', fontcolor=self.__edge_config['font_color'])
        cluster.attr('node', fillcolor=self.__edge_config['color'])

        for edge_num in range(ir_graph.EdgesLength()):
            ir_edge = ir_graph.Edges(edge_num)

            if ir_edge.Type() == IR.Type.EdgeType.EdgeType.DATA:

                # create a node in edge to connect edge blobs
                edge_name = labeler.edge_name(ir_graph.Id(), ir_edge.Id())
                edge_label = labeler.html_str(labeler.data_edge_label(ir_edge.Type()))
                ir_edge_title = labeler.ir_edge_title(ir_edge, self.__hide_edge_names)

                # show edge instructions
                if not self.__hide_instr and ir_edge.HwInfo():
                    # load start step
                    load_start_step = ir_edge.HwInfo().LoadStartStep()
                    load_start_num = 0
                    if load_start_step:
                        step_id = load_start_step.Id()
                        load_start_num = self.__execution_order(ir_graph, step_id)

                    # load sync step
                    load_sync_step = ir_edge.HwInfo().LoadSyncStep()
                    load_sync_num = 0
                    if load_sync_step:
                        step_id = load_sync_step.Id()
                        load_sync_num = self.__execution_order(ir_graph, step_id)

                    # store start step
                    store_start_step = ir_edge.HwInfo().StoreStartStep()
                    store_start_num = 0
                    if store_start_step:
                        step_id = store_start_step.Id()
                        store_start_num = self.__execution_order(ir_graph, step_id)

                    # store sync step
                    store_sync_step = ir_edge.HwInfo().StoreSyncStep()
                    store_sync_num = 0
                    if store_sync_step:
                        step_id = store_sync_step.Id()
                        store_sync_num = self.__execution_order(ir_graph, step_id)

                    instr_label = labeler.edge_instrs_label(load_start_step, load_start_num,
                                                            load_sync_step, load_sync_num,
                                                            store_start_step, store_start_num,
                                                            store_sync_step, store_sync_num)
                    edge_data = [ir_edge_title, instr_label]
                else:
                    edge_data = [ir_edge_title]

                cluster.node(edge_name, labeler.html_str(labeler.list_table(edge_data)),
                             shape='box', style="rounded, bold, filled")

                blob_id = ir_edge.BlobId()
                self.__append_blob_dict(blob_id, blob_type='edge', dst=edge_name,
                                        label=edge_label)

                src_node_name = labeler.node_name(ir_graph.Id(), ir_edge.SrcNodeId())
                dst_node_name = labeler.node_name(ir_graph.Id(), ir_edge.DstNodeId())
                if ir_edge.SrcNodeId() == -1:
                    # create head and connect
                    src_node_name = labeler.head_node_name(ir_graph.Id())
                    cluster.node(src_node_name, shape='point')
                elif ir_edge.DstNodeId() == -1:
                    # create tail and connect
                    dst_node_name = labeler.tail_node_name(ir_graph.Id())
                    cluster.node(dst_node_name, shape='point')

                cluster.edge(src_node_name, edge_name, dir='none')
                cluster.edge(edge_name, dst_node_name)

            elif ir_edge.Type() == IR.Type.EdgeType.EdgeType.CONTROL:
                # TODO: Schema is not yet defined
                cluster.edge(labeler.node_name(ir_graph.Id(), ir_edge.SrcNodeId()),
                             labeler.node_name(ir_graph.Id(), ir_edge.DstNodeId()))

    def __draw_blobs(self, ir_graph: IR.Graph.Graph, cluster: Digraph):
        def blob_iter(ir_graph: IR.Graph.Graph, cluster: Digraph, blobs: list, edge_style: str, edge_dir: str):
            for blob_item in blobs:
                blob = blob_item['blob']
                val = blob_item['val']
                blob_node_name = labeler.blob_node_name(ir_graph.Id(), blob.Id())
                blob_node_label = labeler.blob_node_label(blob)
                blob_node_label = labeler.html_str(blob_node_label)
                # Draw
                cluster.node(blob_node_name, blob_node_label,
                             shape='box', style="rounded, filled")
                if val['blob_type'] == 'lut':
                    # currently do not draw edges for lut blobs
                    pass
                else:
                    cluster.edge(blob_node_name, val['dst'], label=val['edge_label'], style=edge_style, dir=edge_dir)

                if val['blob_type'] == 'edge' and self.__rank_dir == 'LR':
                    with cluster.subgraph() as sub:
                        sub.attr(rank='same')
                        sub.node(val['dst'])
                        sub.node(blob_node_name)

        kernel_blobs = []
        bias_blobs = []
        edge_blobs = []
        lut_blobs = []
        weight_blobs = []       # For LSTM
        
        for blob_num in range(ir_graph.BlobsLength()):
            blob = ir_graph.Blobs(blob_num)
            # check if blob is unused
            if blob.Id() not in self.__blob_dict:
                continue

            for val in self.__blob_dict[blob.Id()]:
                if val['blob_type'] == 'edge':
                    # MUST put EDGE BLOB last to Draw with [rank=same]
                    edge_blobs.append({'blob': blob, 'val': val})
                elif val['blob_type'] == 'kernel':
                    kernel_blobs.append({'blob': blob, 'val': val})
                elif val['blob_type'] == 'bias':
                    bias_blobs.append({'blob': blob, 'val': val})
                elif val['blob_type'] == 'lut':
                    lut_blobs.append({'blob': blob, 'val': val})
                elif val['blob_type'] == 'weight':                      # For LSTM
                    weight_blobs.append({'blob': blob, 'val': val})

        cluster.attr('node', fontsize=self.__kernel_blob_config['font_size'])
        cluster.attr('node', fontcolor=self.__kernel_blob_config['font_color'])
        cluster.attr('node', fillcolor=self.__kernel_blob_config['color'])
        blob_iter(ir_graph, cluster, kernel_blobs, 'dashed', 'none')

        cluster.attr('node', fontsize=self.__bias_blob_config['font_size'])
        cluster.attr('node', fontcolor=self.__bias_blob_config['font_color'])
        cluster.attr('node', fillcolor=self.__bias_blob_config['color'])
        blob_iter(ir_graph, cluster, bias_blobs, 'dashed', 'none')

        cluster.attr('node', fontsize=self.__edge_blob_config['font_size'])
        cluster.attr('node', fontcolor=self.__edge_blob_config['font_color'])
        cluster.attr('node', fillcolor=self.__edge_blob_config['color'])
        blob_iter(ir_graph, cluster, edge_blobs, 'dashed', 'none')

        cluster.attr('node', fontsize=self.__lut_blob_config['font_size'])
        cluster.attr('node', fontcolor=self.__lut_blob_config['font_color'])
        cluster.attr('node', fillcolor=self.__lut_blob_config['color'])
        blob_iter(ir_graph, cluster, lut_blobs, 'none', 'none')

        # For LSTM's weight
        cluster.attr('node', fontsize=self.__kernel_blob_config['font_size'])
        cluster.attr('node', fontcolor=self.__kernel_blob_config['font_color'])
        cluster.attr('node', fillcolor=self.__kernel_blob_config['color'])
        blob_iter(ir_graph, cluster, weight_blobs, 'dashed', 'none')

    def __append_blob_dict(self, blob_id: int, blob_type: str, dst: str, label: str):
        if blob_id not in self.__blob_dict:
            self.__blob_dict[blob_id] = []
        self.__blob_dict[blob_id].append({'blob_type': blob_type, 'dst': dst, 'edge_label': label})

    def __execution_order(self, ir_graph: IR.Graph.Graph, step_id: int):
        # find step position in order array
        if ir_graph.HwInfo():
            order = ir_graph.HwInfo().ExecutionOrderAsNumpy()
            idx = np.where(order == step_id)
            if len(idx[0]) == 1:
                return idx[0][0] + 1
        return 0

    def render(self):
        # generate *.gv file and render graph
        output_file = os.path.splitext(self.input_file)[0] + '.gv'
        dt1 = datetime.datetime.now()
        print('Start at {}'.format(dt1))
        self.p_graph.render(output_file)

        dt2 = datetime.datetime.now()
        print('"{}" is Created with "{}" Engine'
              .format(output_file + '.' + self.__file_format, self.__engine))
        print('Done at {} (Time Elapsed {})'
              .format(dt2, dt2 - dt1))


def parse_input_arguments():
    parser = argparse.ArgumentParser()

    node_group = parser.add_argument_group("node")
    node_group.add_argument("-nfs", "--node-font-size",
                            default=10, help="Set font size of Edge")
    node_group.add_argument("-nfc", "--node-font-color",
                            default='black', help="Set font color of Edge")

    edge_group = parser.add_argument_group("edge")
    edge_group.add_argument("-efs", "--edge-font-size",
                            default=10, help="Set font size of Edge")
    edge_group.add_argument("-efc", "--edge-font-color",
                            default='black', help="Set font color of Edge")
    edge_group.add_argument("-ec", "--edge-color",
                            default='white', help="Set fill color of Edge")

    kernel_blob_group = parser.add_argument_group("kernel_blob")
    kernel_blob_group.add_argument(
        "-kbfs", "--kernel-blob-font-size", default=9, help="Set font size of Kernel Blob")
    kernel_blob_group.add_argument(
        "-kbfc", "--kernel-blob-font-color", default='white', help="Set font color of Kernel Blob")
    kernel_blob_group.add_argument(
        "-kbc", "--kernel-blob-color", default='maroon4', help="Set fill color of Kernel Blob")

    bias_blob_group = parser.add_argument_group("bias_blob")
    bias_blob_group.add_argument(
        "-bbfs", "--bias-blob-font-size", default=9, help="Set font size of Bias Blob")
    bias_blob_group.add_argument(
        "-bbfc", "--bias-blob-font-color", default='white', help="Set font color of Bias Blob")
    bias_blob_group.add_argument(
        "-bbc", "--bias-blob-color", default='royalblue4', help="Set fill color of Bias Blob")

    edge_blob_group = parser.add_argument_group("edge_blob")
    edge_blob_group.add_argument(
        "-ebfs", "--edge-blob-font-size", default=9, help="Edge Blob's font size")
    edge_blob_group.add_argument(
        "-ebfc", "--edge-blob-font-color", default='white', help="Set font color of Edge Blob")
    edge_blob_group.add_argument(
        "-ebc", "--edge-blob-color", default='black', help="Set fill color of Edge Blob")

    lut_blob_group = parser.add_argument_group("lut_blob")
    lut_blob_group.add_argument(
        "-lbfs", "--lut-blob-font-size", default=9, help="LUT Blob's font size")
    lut_blob_group.add_argument(
        "-lbfc", "--lut-blob-font-color", default='black', help="Set font color of LUT Blob")
    lut_blob_group.add_argument(
        "-lbc", "--lut-blob-color", default='powderblue', help="Set fill color of LUT Blob")

    mem_info_group = parser.add_argument_group("mem_info")
    mem_info_group.add_argument(
        "-mifs", "--mem-info-font-size", default=9, help="Set font size of Memory Info")
    mem_info_group.add_argument(
        "-mifc", "--mem-info-font-color", default='white', help="Set font color of Memory Info")
    mem_info_group.add_argument(
        "-mic", "--mem-info-color", default='turquoise4', help="Set fill color of Memory Info")

    parser.add_argument("-e", "--engine", default='dot',
                        help="Graph rendering engine")
    parser.add_argument("-fr", "--format", default='svg',
                        help="Visualizer output file format")
    parser.add_argument("-rd", "--rank-dir", default='TB',
                        help="Graph Direction Left-Right / Top-Bottom")
    parser.add_argument("-hb", "--hide-blobs", default='false',
                        help="If set true, All blobs(kernel, bias, edge) will not be rendered")
    parser.add_argument("-he", "--hide-edge-names", default='false',
                        help="If set true, All edge names will not be rendered")
    parser.add_argument("-hi", "--hide-instr", default='false',
                        help="If set true, All instructions will not be rendered")
    parser.add_argument("-hm", "--hide-meminfos", default='false', help="Show/Hide Memory Infos")
    parser.add_argument("-i", "--ir", help="Input ir file name")

    args = parser.parse_args()
    return args


def verify_args(args: str):
    if not args.ir:
        sys.exit()

    if(args.rank_dir not in ('TB', 'LR')):
        print('Invalid argument {}: Input LR or TB'.format(args.rank_dir))
        sys.exit()

    if(args.format not in ('svg', 'png')):
        print('Invalid argument {}: Input svg or png'.format(args.format))
        sys.exit()

    if(args.hide_blobs not in ('true', 'false')):
        print('Invalid argument {}: Input true or false'.format(args.hide_blobs))
        sys.exit()

    if(args.hide_edge_names not in ('true', 'false')):
        print('Invalid argument {}: Input true or false'.format(args.hide_edge_names))
        sys.exit()

    if(args.hide_instr not in ('true', 'false')):
        print('Invalid argument {}: Input true or false'.format(args.hide_instr))
        sys.exit()

    if(args.hide_meminfos not in ('true', 'false')):
        print('Invalid argument {}: Input true or false'.format(args.hide_meminfos))
        sys.exit()


if __name__ == "__main__":
    # process arguments
    args = parse_input_arguments()
    verify_args(args)

    input_file = args.ir

    ir_visualizer = Visualizer(input_file, args)
    ir_visualizer.show_ir_graph_info()

    ir_visualizer.draw()
    ir_visualizer.render()
