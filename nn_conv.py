import torch
import torch.nn.functional as F

input =torch.tensor([[1,2,0,2,1],
                     [2,3,4,5,6],
                     [1,6,4,2,1],
                     [3,4,5,2,1],
                     [4,5,2,1,5]])
kernel=torch.tensor([[1,2,4],
                     [2,4,5],
                     [2,6,8]])
input=torch.reshape(input,(1,1,5,5)) #样本数，矩阵深度，高度，宽度
kernel=torch.reshape(kernel,(1,1,3,3))

print(input.shape)
print(kernel.shape)

# 卷积函数
output=F.conv2d(input,kernel,stride=1)
print(output)

output2=F.conv2d(input,kernel,stride=2)
print(output2)

output3=F.conv2d(input,kernel,stride=1,padding=1)
print(output3)