import torch
import torchvision
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset=torchvision.datasets.CIFAR10("./dataset_CIFAR10",train=False,transform=torchvision.transforms.ToTensor(),download=True)

dataloader=DataLoader(dataset,batch_size=64)

class MyNN(nn.Module):
    def __init__(self):
        super(MyNN,self).__init__()
        self.maxpool=MaxPool2d(kernel_size=3,ceil_mode=False)

    def forward(self,input):
        output=self.maxpool(input)
        return output
nn_pool=MyNN()
writer=SummaryWriter("./CIFAR10_logs")
step=0
for data in dataloader:
    imgs,targets=data
    writer.add_images("最大池化层input",imgs,step)
    output=nn_pool(imgs)
    writer.add_images("最大池化层output",output,step)
    step+=1

writer.close()
