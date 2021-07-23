# -*- coding: utf-8 -*-
import torch
import sys
import os
from ir_net.utils import Nnrt
from ir_net.utils import inference_pytorch, inference_nnruntime, compare_tensors
current_dir = os.path.dirname(os.path.abspath(__file__))


def test_simple_add_net(graph_ir_file : str):
    from ir_net.simple_add import TestNet
    # Set inputs
    x1 = torch.tensor([[1,2,3],[4,5,6]])
    x2 = torch.tensor([[10,20,20],[40,50,60]])
    # inference
    pytorch_outs = inference_pytorch(TestNet(), [x1, x2])
    nnrt_outs = inference_nnruntime(graph_ir_file, [x1, x2])
    # compare
    assert compare_tensors([pytorch_outs], nnrt_outs)

def test_simple_if_net(graph_ir_file : str):
    from ir_net.simple_net_if import TestNet3
    # Set inputs
    x1 = torch.tensor([[1, 3, 5], [9, 7, 5]])
    x2 = torch.tensor([[10, 30, 50], [90, -20, 10]])
    x3 = torch.tensor(18)
    x4 = torch.tensor(7)
    # inference
    pytorch_outs = inference_pytorch(TestNet3(), [x1, x2, x3, x4])
    nnrt_outs = inference_nnruntime(graph_ir_file, [x1, x2, x3, x4])
    # compare
    assert compare_tensors([pytorch_outs], nnrt_outs)


def test_simple_loop_net(graph_ir_files : str):
    from ir_net.simple_loop_net import TestNet1, TestNet2
    # Set inputs
    x1 = torch.tensor([[1,2,3],[4,5,6]])
    x2 = torch.tensor([[20,2,30],[-4,50,70]])
    # inference
    models = [TestNet1(), TestNet2()]
    for ir_file, model in zip(graph_ir_files, models):
        pytorch_outs = inference_pytorch(model, [x1, x2])
        nnrt_outs = inference_nnruntime(ir_file, [x1, x2])
        # compare
        assert compare_tensors([pytorch_outs], nnrt_outs)


if __name__ == '__main__':
    # Set IR files
    add_ir_file = os.path.join(current_dir, 'ir_files/simple_add_net/sample_add_frontend.ir')
    loop_ir_files = [
        os.path.join(current_dir, 'ir_files/simple_loop_net/loop_frontend_1.ir'),
        os.path.join(current_dir, 'ir_files/simple_loop_net/loop_frontend_2.ir')
    ]
    if_ir_file = os.path.join(current_dir, 'ir_files/simple_if_net/if_frontend_3.ir')

    # inference & compare result
    # test_simple_add_net(add_ir_file)
    test_simple_loop_net(loop_ir_files)
