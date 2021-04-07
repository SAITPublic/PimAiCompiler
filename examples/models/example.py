import torch

class MyDecisionGate(torch.nn.Module):
    def forward(self, x):
        if x.sum() > 0:
            return x
        else:
            return -x

class MyCell(torch.nn.Module):
    def __init__(self, dg):
        super(MyCell, self).__init__()
        self.dg = dg
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, x, h):
        new_h = torch.tanh(self.dg(self.linear(x)) + h)
        return new_h, new_h

scripted_gate = torch.jit.script(MyDecisionGate())
my_cell = MyCell(scripted_gate)
x, h = torch.rand(3, 4), torch.rand(3, 4)
scripted_cell = torch.jit.script(my_cell, (x, h))
#scripted_cell.save('my_cell.pt')
print(scripted_cell.graph)
