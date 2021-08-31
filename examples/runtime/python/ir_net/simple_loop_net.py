import torch

class TestNet1(torch.nn.Module):
    def __init__(self):
        super(TestNet1, self).__init__()

    def forward(self, x1, x2):
        y1 = x1
        for i in range(3):
            y1 = torch.add(y1, x2)
            y1 = torch.add(y1, i)
        y1 = torch.add(y1, 25)
        return y1


class TestNet2(torch.nn.Module):
    def __init__(self):
        super(TestNet2, self).__init__()

    def forward(self, x1, x2):
        y1 = x1
        for i in range(3):
            for j in range(2):
                y1 = torch.add(y1, x2)
                y1 = torch.add(y1, i)
        y1 = torch.add(y1, 25)
        return y1


def simple1():
    x1 = torch.tensor([[1,2,3],[4,5,6]])
    x2 = torch.tensor([[20,2,30],[-4,50,70]])
    model = TestNet1()
    model.eval()
    output = model(x1, x2)
    script_model = torch.jit.script(model)
    frozen_model = torch.jit.freeze(script_model)
    print(frozen_model.graph)
    print('x1:{}'.format(x1))
    print('x2:{}'.format(x2))
    print('out:{}'.format(output))
    frozen_model.save("simple_loop1.pt")


def simple2():
    x1 = torch.tensor([[1,2,3],[4,5,6]])
    x2 = torch.tensor([[20,2,30],[-4,50,70]])
    model = TestNet2()
    model.eval()
    output = model(x1, x2)
    script_model = torch.jit.script(model)
    frozen_model = torch.jit.freeze(script_model)
    print(frozen_model.graph)
    print('x1:{}'.format(x1))
    print('x2:{}'.format(x2))
    print('out:{}'.format(output))
    frozen_model.save("simple_loop2.pt")


if __name__ == '__main__':
    simple2()
