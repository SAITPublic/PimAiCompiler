# -*- coding: utf-8 -*-

import torch
import sys
# Load Nnr module
sys.path.append('/opt/rocm/lib')
import Nnrt


if __name__ == '__main__':
    # help(Nnrt)
    rt = Nnrt.NNRuntime("/home/rnnt.torchscript")
    rt.test()

    inputs = [torch.randn(5, 5) for i in range(3)]
    outputs = rt.inferenceModel(input_tensors=inputs)

    print(len(outputs))
    for item in outputs:
        print('shape:{} dtype:{} device:{}'.format(item.size(), item.dtype, item.device))
    #n.inference()
