import torch
import torch.nn as nn

# if -- else
def sample1():
    class TestNet(nn.Module):
        def __init__(self):
            super(TestNet, self).__init__()

        def forward(self, x1 : torch.Tensor, x2 : torch.Tensor, x3 : int):
            y = torch.add(x1, 20)
            y = torch.add(x2, y)
            z1 = x3 + 3
            if z1 > 10:
                y = torch.add(y, 15)
            else:
                y = torch.add(y, -15)
            return y
        
    net = TestNet()
    net.eval()
    x1 = torch.tensor([[1, 3, 5], [9, 7, 5]])
    x2 = torch.tensor([[10, 30, 50], [90, -20, 10]])
    out = net(x1, x2, 8)

    script_net = torch.jit.script(net)
    frozen_net = torch.jit.freeze(script_net)

    print('='*30)
    print(frozen_net.graph)
    print('='*30)
    print('x1:{}'.format(x1))
    print('x2:{}'.format(x2))
    print('out:{}'.format(out))

# graph(%self : __torch__.___torch_mangle_0.TestNet,
#       %x1.1 : Tensor,
#       %x2.1 : Tensor,
#       %x3.1 : int):
#   %9 : int = prim::Constant[value=15]() 
#   %8 : int = prim::Constant[value=10]() 
#   %7 : int = prim::Constant[value=3]() 
#   %6 : int = prim::Constant[value=20]() 
#   %5 : int = prim::Constant[value=1]()
#   %4 : int = prim::Constant[value=-15]() 
#   %y.1 : Tensor = aten::add(%x1.1, %6, %5) 
#   %y.3 : Tensor = aten::add(%x2.1, %y.1, %5) 
#   %z1.1 : int = aten::add(%x3.1, %7) 
#   %13 : bool = aten::gt(%z1.1, %8) 
#   %y : Tensor = prim::If(%13)
#     block0():
#       %y.5 : Tensor = aten::add(%y.3, %9, %5) 
#       -> (%y.5)
#     block1():
#       %y.8 : Tensor = aten::add(%y.3, %4, %5) 
#       -> (%y.8)
#   return (%y)

# ==============================
# x1:tensor([[1, 3, 5],
#         [9, 7, 5]])
# x2:tensor([[ 10,  30,  50],
#         [ 90, -20,  10]])
# out:tensor([[ 46,  68,  90],
#         [134,  22,  50]])


# sample2: only if, no else
def sample2():
    class TestNet(nn.Module):
        def __init__(self):
            super(TestNet, self).__init__()

        def forward(self, x1 : torch.Tensor, x2 : torch.Tensor, x3 : int):
            y = torch.add(x1, 20)
            y = torch.add(x2, y)
            z1 = x3 + 3
            if z1 > 10:
                y = torch.add(y, 15)
                y = torch.add(y, 35)
            y = torch.add(y, 100)
            y = torch.add(y, z1)
            return y

        
    net = TestNet()
    net.eval()
    x1 = torch.tensor([[1, 3, 5], [9, 7, 5]])
    x2 = torch.tensor([[10, 30, 50], [90, -20, 10]])
    out = net(x1, x2, 8)

    script_net = torch.jit.script(net)
    frozen_net = torch.jit.freeze(script_net)

    print('='*30)
    print(frozen_net.graph)
    print('='*30)
    print('x1:{}'.format(x1))
    print('x2:{}'.format(x2))
    print('out:{}'.format(out))

# ==============================
# graph(%self : __torch__.___torch_mangle_0.TestNet,
#       %x1.1 : Tensor,
#       %x2.1 : Tensor,
#       %x3.1 : int):
#   %10 : int = prim::Constant[value=100]() 
#   %9 : int = prim::Constant[value=35]() 
#   %8 : int = prim::Constant[value=15]() 
#   %7 : int = prim::Constant[value=10]() 
#   %6 : int = prim::Constant[value=3]() 
#   %5 : int = prim::Constant[value=20]() 
#   %4 : int = prim::Constant[value=1]()
#   %y.1 : Tensor = aten::add(%x1.1, %5, %4) 
#   %y.3 : Tensor = aten::add(%x2.1, %y.1, %4) 
#   %z1.1 : int = aten::add(%x3.1, %6) 
#   %14 : bool = aten::gt(%z1.1, %7) 
#   %y : Tensor = prim::If(%14) 
#     block0():
#       %y.5 : Tensor = aten::add(%y.3, %8, %4) 
#       %y.8 : Tensor = aten::add(%y.5, %9, %4) 
#       -> (%y.8)
#     block1():
#       -> (%y.3)
#   %y.14 : Tensor = aten::add(%y, %10, %4) 
#   %y.16 : Tensor = aten::add(%y.14, %z1.1, %4) 
#   return (%y.16)

# ==============================
# x1:tensor([[1, 3, 5],
#         [9, 7, 5]])
# x2:tensor([[ 10,  30,  50],
#         [ 90, -20,  10]])
# out:tensor([[192, 214, 236],
#         [280, 168, 196]])

# if -- if--else -- else --
def sample3():
    class TestNet(nn.Module):
        def __init__(self):
            super(TestNet, self).__init__()

        def forward(self, x1 : torch.Tensor, x2 : torch.Tensor, x3 : int):
            y = torch.add(x1, 20)
            y = torch.add(x2, y)
            z1 = x3 + 3
            z2 = x3 + 8
            if z1 > 10:
                y = torch.add(y, 15)
                if z2 > 10:
                    y = torch.add(y, 25)
                else:
                    y = torch.add(y, -25)
            else:
                y = torch.add(y, 100)
            y = torch.add(y, z1)
            return y

        
    net = TestNet()
    net.eval()
    x1 = torch.tensor([[1, 3, 5], [9, 7, 5]])
    x2 = torch.tensor([[10, 30, 50], [90, -20, 10]])
    out = net(x1, x2, 8)

    script_net = torch.jit.script(net)
    frozen_net = torch.jit.freeze(script_net)

    print('='*30)
    print(frozen_net.graph)
    print('='*30)
    print('x1:{}'.format(x1))
    print('x2:{}'.format(x2))
    print('out:{}'.format(out))


# ==============================
# graph(%self : __torch__.___torch_mangle_0.TestNet,
#       %x1.1 : Tensor,
#       %x2.1 : Tensor,
#       %x3.1 : int):
#   %12 : int = prim::Constant[value=100]() 
#   %11 : int = prim::Constant[value=25]() 
#   %10 : int = prim::Constant[value=15]() 
#   %9 : int = prim::Constant[value=10]() 
#   %8 : int = prim::Constant[value=8]() 
#   %7 : int = prim::Constant[value=3]() 
#   %6 : int = prim::Constant[value=20]() 
#   %5 : int = prim::Constant[value=1]()
#   %4 : int = prim::Constant[value=-25]() 
#   %y.1 : Tensor = aten::add(%x1.1, %6, %5) 
#   %y.3 : Tensor = aten::add(%x2.1, %y.1, %5) 
#   %z1.1 : int = aten::add(%x3.1, %7) 
#   %z2.1 : int = aten::add(%x3.1, %8) 
#   %17 : bool = aten::gt(%z1.1, %9) 
#   %y : Tensor = prim::If(%17) 
#     block0():
#       %y.5 : Tensor = aten::add(%y.3, %10, %5) 
#       %20 : bool = aten::gt(%z2.1, %9) 
#       %y.27 : Tensor = prim::If(%20) 
#         block0():
#           %y.8 : Tensor = aten::add(%y.5, %11, %5) 
#           -> (%y.8)
#         block1():
#           %y.11 : Tensor = aten::add(%y.5, %4, %5) 
#           -> (%y.11)
#       -> (%y.27)
#     block1():
#       %y.18 : Tensor = aten::add(%y.3, %12, %5) 
#       -> (%y.18)
#   %y.25 : Tensor = aten::add(%y, %z1.1, %5) 
#   return (%y.25)

# ==============================
# x1:tensor([[1, 3, 5],
#         [9, 7, 5]])
# x2:tensor([[ 10,  30,  50],
#         [ 90, -20,  10]])
# out:tensor([[ 82, 104, 126],
#         [170,  58,  86]])



if __name__ == '__main__':
    sample3()
