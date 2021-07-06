import torch
import torch.nn as nn

# if -- else
def sample1():
    class TestNet(nn.Module):
        def __init__(self):
            super(TestNet, self).__init__()

        def forward(self, x1 : torch.Tensor, x2 : torch.Tensor, x3 : torch.Tensor):
            y = torch.add(x1, 20)
            y = torch.add(x2, y)
            z1 = x3.item()
            if z1 > 10:
                y = torch.add(y, 15)
            else:
                y = torch.add(y, -15)
            return y
        
    net = TestNet()
    net.eval()
    x1 = torch.tensor([[1, 3, 5], [9, 7, 5]])
    x2 = torch.tensor([[10, 30, 50], [90, -20, 10]])
    out = net(x1, x2, torch.tensor(8))

    script_net = torch.jit.script(net)
    frozen_net = torch.jit.freeze(script_net)

    frozen_net.save('sample_if_else_1.pt')

    print('='*30)
    print(frozen_net.graph)
    print('='*30)
    print('x1:{}'.format(x1))
    print('x2:{}'.format(x2))
    print('out:{}'.format(out))


# sample2: only if, no else
def sample2():
    class TestNet(nn.Module):
        def __init__(self):
            super(TestNet, self).__init__()

        def forward(self, x1 : torch.Tensor, x2 : torch.Tensor, x3 : torch.Tensor):
            y = torch.add(x1, 20)
            y = torch.add(x2, y)
            z1 = x3.item()
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
    out = net(x1, x2, torch.tensor(8))

    script_net = torch.jit.script(net)
    frozen_net = torch.jit.freeze(script_net)

    print('='*30)
    print(frozen_net.graph)
    print('='*30)
    print('x1:{}'.format(x1))
    print('x2:{}'.format(x2))
    print('out:{}'.format(out))
    frozen_net.save('sample_if_2.pt')


# if -- if--else -- else --
def sample3():
    class TestNet(nn.Module):
        def __init__(self):
            super(TestNet, self).__init__()

        def forward(self, x1 : torch.Tensor, x2 : torch.Tensor, x3 : torch.Tensor, x4 : torch.Tensor):
            y = torch.add(x1, 20)
            y = torch.add(x2, y)
            z1 = x3.item()
            z2 = x4.item()
            if z1 > 10:
                y = torch.add(y, 15)
                if z2 > 5:
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
    out = net(x1, x2, torch.tensor(18), torch.tensor(7))

    script_net = torch.jit.script(net)
    frozen_net = torch.jit.freeze(script_net)

    print('='*30)
    print(frozen_net.graph)
    print('='*30)
    print('x1:{}'.format(x1))
    print('x2:{}'.format(x2))
    print('out:{}'.format(out))
    frozen_net.save('sample_if_if_else_else_3.pt')


if __name__ == '__main__':
    sample3()
