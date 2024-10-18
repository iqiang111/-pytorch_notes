import torchvision
from torch import nn
from torch.nn import ReLU, Sigmoid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset=torchvision.datasets.CIFAR10("./dataset_CIFAR10",train=False,transform=torchvision.transforms.ToTensor(),download=True)

dataloader=DataLoader(dataset,batch_size=64)

class MyNN(nn.Module):
    def __init__(self):
        super(MyNN,self).__init__()
        self.relu=ReLU()
        self.sigmoid=Sigmoid()

    def forward(self,input):
        output=self.relu(input)
        return output

nn_relu=MyNN()
writer=SummaryWriter("./CIFAR10_logs")
step=0
for data in dataloader:
    imgs,targets=data
    # writer.add_images("非新型变换input",imgs,step)
    output=nn_relu(imgs)
    writer.add_images("非线型变换output+sigmoid+relu",output,step)
    step+=1

writer.close()