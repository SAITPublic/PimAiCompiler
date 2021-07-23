import torch
import os
import shutil
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
nnrt_path = os.path.abspath(os.path.join(current_dir, '../../../build/runtime/python'))
# Load Nnr module
sys.path.append(nnrt_path)
import Nnrt


def script_freeze(torch_model):
    torch_model.eval()
    script_model = torch.jit.script(torch_model)
    frozen_model = torch.jit.freeze(script_model)
    return frozen_model

def inference_pytorch(torch_model, inputs):
    torch_model.eval()
    outs = torch_model(*inputs)
    return outs

def inference_nnruntime(graph_ir_file, inputs):
    rt = Nnrt.NNRuntime(graph_ir_file)
    outs = rt.inferenceModel(inputs)
    return outs

def compare_tensors(lst1, lst2):
    for t1, t2 in zip(lst1, lst2):
        print(t1)
        print(t2)
        if not t1.equal(t2):
            return False
    return True
