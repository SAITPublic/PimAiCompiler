# -*- coding: utf-8 -*-
import argparse
from typing import Mapping
import torch
import sys
import os
import time
import timeit
import tqdm
from ir_net.utils import NNCompiler
from ir_net.utils import inference_pytorch, inference_nncompiler, compare_tensors
current_dir = os.path.dirname(os.path.abspath(__file__))


def test_simple_add_net(graph_file : str):
    from ir_net.simple_add import TestNet
    # Set inputs
    x1 = torch.tensor([[1,2,3],[4,5,6]])
    x2 = torch.tensor([[10,20,20],[40,50,60]])
    # inference
    pytorch_outs = inference_pytorch(TestNet(), [x1, x2])
    nncompiler_outs = inference_nncompiler(graph_file, [x1, x2])
    # compare
    assert compare_tensors([pytorch_outs], nncompiler_outs)

def test_simple_if_net(graph_file : str):
    from ir_net.simple_net_if import TestNet3
    # Set inputs
    x1 = torch.tensor([[1, 3, 5], [9, 7, 5]])
    x2 = torch.tensor([[10, 30, 50], [90, -20, 10]])
    x3 = torch.tensor(18)
    x4 = torch.tensor(7)
    # inference
    pytorch_outs = inference_pytorch(TestNet3(), [x1, x2, x3, x4])
    nncompiler_outs = inference_nncompiler(graph_file, [x1, x2, x3, x4])
    # compare
    assert compare_tensors([pytorch_outs], nncompiler_outs)


def test_simple_loop_net(graph_files : str):
    from ir_net.simple_loop_net import TestNet1, TestNet2
    # Set inputs
    x1 = torch.tensor([[1,2,3],[4,5,6]])
    x2 = torch.tensor([[20,2,30],[-4,50,70]])
    # inference
    models = [TestNet1(), TestNet2()]
    for ir_file, model in zip(graph_files, models):
        pytorch_outs = inference_pytorch(model, [x1, x2])
        nncompiler_outs = inference_nncompiler(ir_file, [x1, x2])
        # compare
        assert compare_tensors([pytorch_outs], nncompiler_outs)

def test_simple_cases():
    # Some simple cases
    # Set graph files
    add_file = os.path.join(current_dir, '../resource/graph_files/simple_add_net/simple_jit_add.torchscript')
    loop_files = [
        os.path.join(current_dir, '../resource/graph_files/simple_loop_net/simple_loop1.torchscript'),
        os.path.join(current_dir, '../resource/graph_files/simple_loop_net/simple_loop2.torchscript')
    ]
    if_file = os.path.join(current_dir, '../resource/graph_files/simple_if_net/sample_if_if_else_else_3.torchscript')

    # inference & compare result
    test_simple_add_net(add_file)
    test_simple_loop_net(loop_files)
    test_simple_if_net(if_file)


def test_rnnt_inference(input_file : str, feature_file : str, feature_len_file : str, model_type : str):
    # Prepare inputs
    feature = torch.load(feature_file).cuda()   # dtype=torch.fp16
    feature_len = torch.load(feature_len_file)  # dtype=torch.long
    # Init nncompiler
    nncompiler = NNCompiler.PipelineManager(input_file, model_type)
    # warn-up
    _, _, _ = nncompiler.inferenceModel([feature, feature_len])
    # Run and test
    time_start = time.time()
    test_cnt = 100
    for _ in tqdm.tqdm(range(test_cnt)):
        _, _, transcript = nncompiler.inferenceModel([feature, feature_len])
    time_end = time.time()
    print('RNNT avg_time:{}ms'.format((time_end - time_start) / test_cnt * 1000))


def test_hwr_inference(input_file : str, input_tensor_file : str, model_type : str):
    # Prepare inputs
    input_tensor = torch.load(input_tensor_file).cuda()   # dtype=torch.fp16
    # Init nncompiler
    nncompiler = NNCompiler.PipelineManager(input_file, model_type)
    # warn-up
    _ = nncompiler.inferenceModel([input_tensor])
    # Run and test
    torch.cuda.synchronize()
    time_start = time.time()
    test_cnt = 100
    for _ in tqdm.tqdm(range(test_cnt)):
        outpus = nncompiler.inferenceModel([input_tensor])
        torch.cuda.synchronize()
    time_end = time.time()
    print('HWR avg_time:{}ms'.format((time_end - time_start) / test_cnt * 1000))


def test_gnmt_inference(input_file : str, src_file : str, src_length_file : str,  bos_file : str, model_type : str):
    # Prepare inputs
    src = torch.load(src_file).cuda()   # dtype=torch.long
    src_length = torch.load(src_length_file).cuda()   # dtype=torch.long
    bos = torch.load(bos_file).cuda()   # dtype=torch.long
    # Init nncompiler
    nncompiler = NNCompiler.PipelineManager(input_file, model_type)
        # warn-up
    _, _, _ = nncompiler.inferenceModel([src, src_length, bos])
    # Run and test
    time_start = time.time()
    test_cnt = 100
    for _ in tqdm.tqdm(range(test_cnt)):
        _, _, transcript = nncompiler.inferenceModel([src, src_length, bos])
    time_end = time.time()
    print('GNMT avg_time:{}ms'.format((time_end - time_start) / test_cnt * 1000))


def test_transformer_inference(input_file : str, input_tensor_file : str, model_type : str):
    # Prepare inputs
    input_tensor = torch.load(input_tensor_file).cuda()
    # Init nncompiler
    nncompiler = NNCompiler.PipelineManager(input_file, model_type)
    # warn-up
    _ = nncompiler.inferenceModel([input_tensor])
    # Run and test
    torch.cuda.synchronize()
    time_start = time.time()
    test_cnt = 100
    for _ in tqdm.tqdm(range(test_cnt)):
        outpus = nncompiler.inferenceModel([input_tensor])
        torch.cuda.synchronize()
    time_end = time.time()
    print('Transformer avg_time:{}ms'.format((time_end - time_start) / test_cnt * 1000))


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--model_kind", type=str, choices=['RNNT', 'HWR', 'GNMT', 'Transformer'], help='choose model to inference', required=True)
    arg_parser.add_argument('--input_file', type=str, help='Input file', required=True)
    args = arg_parser.parse_args()
    
    assert os.path.exists(args.input_file)
    if args.model_kind == 'RNNT':
        # Inference RNNT
        rnnt_input_file = args.input_file
        feature_file = os.path.join(current_dir, '../resource/rnnt/inputs/feature.pth')
        feature_len_file = os.path.join(current_dir,'../resource/rnnt/inputs/feature_len.pth')
        assert os.path.exists(feature_file) and os.path.exists(feature_len_file)
        test_rnnt_inference(rnnt_input_file, feature_file, feature_len_file, args.model_kind)
    elif args.model_kind == 'HWR':
        input_tensor_file = os.path.join(current_dir, '../resource/hwr/inputs/input_hwr_1_1_1024_128.pt')
        assert os.path.exists(input_tensor_file)
        test_hwr_inference(args.input_file, input_tensor_file, args.model_kind)
    elif args.model_kind == 'GNMT':
        gnmt_input_file = args.input_file
        src_file = os.path.join(current_dir, '../resource/gnmt/inputs/src_1_12_torch.cuda.LongTensor.pt')
        src_length_file = os.path.join(current_dir, '../resource/gnmt/inputs/src_length_1_torch.cuda.LongTensor.pt')
        bos_file = os.path.join(current_dir, '../resource/gnmt/inputs/bos_1_1_torch.cuda.LongTensor.pt')
        assert os.path.exists(src_file) and os.path.exists(src_length_file) and os.path.exists(bos_file)
        test_gnmt_inference(gnmt_input_file, src_file, src_length_file, bos_file, args.model_kind)
    elif args.model_kind == 'Transformer':
        input_tensor_file = os.path.join(current_dir, '../resource/transformer/inputs/transformer.pt')
        assert os.path.exists(input_tensor_file)
        test_transformer_inference(args.input_file, input_tensor_file, args.model_kind)
