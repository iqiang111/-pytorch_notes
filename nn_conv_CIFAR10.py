import torch
import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset=torchvision.datasets.CIFAR10("./dataset_CIFAR10",train=False,transform=torchvision.transforms.ToTensor(),download=True)

dataloader=DataLoader(dataset,batch_size=64)

class MyNN(nn.Module):
    def __init__(self):
        super(MyNN,self).__init__()
        self.conv1=Conv2d(in_channels=3,out_channels=6,kernel_size=3,stride=1,padding=0)

    def forward(self,x):
        x=self.conv1(x)
        return x

myNN=MyNN()
print(myNN)

#tensorboard
writer=SummaryWriter("./CIFAR10_logs")
step=0
for data in dataloader:
    imgs,targets=data
    output=myNN(imgs)
    # print(imgs.shape)
    # print(output.shape)
    writer.add_images("卷积层输入",imgs,step)

    #转换
    output=torch.reshape(output,(-1,3,30,30))
    writer.add_images("卷积层输出",output,step)
    step+=1


writer.close()