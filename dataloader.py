import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

#测试数据集
test_data=torchvision.datasets.CIFAR10(root="./dataset_CIFAR10",train=False,transform=torchvision.transforms.ToTensor())
#数据加载
test_loader=DataLoader(dataset=test_data,batch_size=64,shuffle=True,num_workers=0,drop_last=False)
#测试数据集中第一张图片及target
img,target=test_data[0]
print(img.shape)
print(target)
#使用 tensorboard 可视化过程
writer=SummaryWriter("CIFAR10_logs")

for epoch in range(2):
    step=0;
    for data in test_loader:
        imgs,targets=data
        writer.add_images("开启混洗:{}".format(epoch),imgs,step)
        step+=1

writer.close()

