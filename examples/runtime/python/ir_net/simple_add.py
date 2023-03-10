# Copyright (C) 2021 Samsung Electronics Co. LTD
# 
# This software is a property of Samsung Electronics.
# No part of this software, either material or conceptual 
# may be copied or distributed, transmitted,transcribed, 
# stored in a retrieval system, or translated into any human 
# or computer language in any form by any means, electronic, 
# mechanical, manual or otherwise, or disclosed to third 
# parties without the express written permission of Samsung 
# Electronics. (Use of the Software is restricted to 
# non-commercial, personal or academic, research purpose only)
# ==============================================================

import torch


# Define Net
class TestNet(torch.nn.Module):
    def __init__(self):
        super(TestNet, self).__init__()

    def forward(self, x1, x2):
        y1 = torch.add(x1, 10)
        y2 = torch.add(x2, 5)
        y3 = torch.add(y1, y2)
        y4 = torch.add(y3, 10)
        return y4


def sample1():
    x1 = torch.tensor([[1,2,3],[4,5,6]])
    x2 = torch.tensor([[10,20,20],[40,50,60]])

    model = TestNet()
    model.eval()
    output = model(x1, x2)
    my_script_module = torch.jit.script(model)
    frozen_model = torch.jit.freeze(my_script_module)
    print(frozen_model.graph)
    torch.jit.save(frozen_model, "simple_jit_add.torchscript")

    print("x1:{}".format(x1))
    print("x2:{}".format(x2))
    print("output:{}".format(output))


if __name__ == '__main__':
    sample1()
