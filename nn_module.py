import torch
from torch import nn


class nnDemo(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,input):
        output=input+1
        return output

demo1=nnDemo()
x=torch.tensor(4)
output=demo1(x)
print(output)