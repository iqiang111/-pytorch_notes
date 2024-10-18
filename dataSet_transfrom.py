import torchvision
from torch.utils.tensorboard import SummaryWriter

dataset_transform=torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()  #PIL  --->  tensor 转换图片类型
])
train_set = torchvision.datasets.CIFAR10(root="./dataset_CIFAR10",transform=dataset_transform,train=True,download=True)
test_set = torchvision.datasets.CIFAR10(root="./dataset_CIFAR10",transform=dataset_transform,train=False,download=True)

data=torchvision.datasets.FakeData

writer=SummaryWriter("CIFAR10_logs")
for i in range(10):
    img,target=test_set[i]
    writer.add_image("test_set",img,i)


writer.close()
# print(train_set[0])
# print(test_set.classes)
#
# img,target=test_set[0]
# print(img)
# print(target)
# print(test_set.classes[target])
# img.show()