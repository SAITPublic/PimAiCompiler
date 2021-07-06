import torch


# add model

def sample1():
    class TestNet(torch.nn.Module):
        def __init__(self):
            super(TestNet, self).__init__()

        def forward(self, a):
            a = torch.add(a, 78)
            return a

    in_a = torch.tensor([[1,2,3],[4,5,6]])
    model = TestNet()
    output = model.forward(in_a)
    my_script_module = torch.jit.script(model)
    print(my_script_module.graph)
    torch.jit.save(my_script_module, "simple_jit_add_1.pt")


def sample2():
    class TestNet(torch.nn.Module):
        def __init__(self):
            super(TestNet, self).__init__()

        def forward(self, x1, x2):
            y1 = torch.add(x1, 10)
            y2 = torch.add(x2, 5)
            y3 = torch.add(y1, y2)
            y4 = torch.add(y3, 10)
            return y4

    x1 = torch.tensor([[1,2,3],[4,5,6]])
    x2 = torch.tensor([[10,20,20],[40,50,60]])

    model = TestNet()
    model.eval()
    output = model(x1, x2)
    my_script_module = torch.jit.script(model)
    frozen_model = torch.jit.freeze(my_script_module)
    print(frozen_model.graph)
    torch.jit.save(frozen_model, "simple_jit_add_2.pt")

    print("x1:{}".format(x1))
    print("x2:{}".format(x2))
    print("output:{}".format(output))

# The outputGraph
# graph(%self : __torch__.___torch_mangle_0.TestNet,
#       %x1.1 : Tensor,
#       %x2.1 : Tensor):
#   %5 : int = prim::Constant[value=5]()
#   %4 : int = prim::Constant[value=10]() 
#   %3 : int = prim::Constant[value=1]()
#   %y1.1 : Tensor = aten::add(%x1.1, %4, %3)
#   %y2.1 : Tensor = aten::add(%x2.1, %5, %3)
#   %y3.1 : Tensor = aten::add(%y1.1, %y2.1, %3)
#   %y4.1 : Tensor = aten::add(%y3.1, %4, %3)
#   return (%y4.1)

# The input & output tensors
# x1:tensor([[1, 2, 3],
#         [4, 5, 6]])
# x2:tensor([[10, 20, 20],
#         [40, 50, 60]])
# output:tensor([[36, 47, 48],
#         [69, 80, 91]])



if __name__ == '__main__':
    sample2()
