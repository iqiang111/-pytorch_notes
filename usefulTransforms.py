from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

img=Image.open("test_img.png")
writer=SummaryWriter("logs")
print(img)

#ToTensor
trans_totensor=transforms.ToTensor()
img_tensor=trans_totensor(img)
writer.add_image("ToTensor",img_tensor)

#Normalize  归一化
print(img_tensor[0][0][0])
trans_norm=transforms.Normalize([1,2,3],[4,2,3])
img_norm=trans_norm(img_tensor)
print(img_norm[0][0][0])
writer.add_image("Normalize",img_norm,4)


#Resize
print(img.size)
trans_resize=transforms.Resize((1024,1024))
#img PIL -> resize -> totensor
img_resize=trans_resize(img)
img_resize=trans_totensor(img_resize)
writer.add_image("Resize",img_resize,1)
print(img_resize)

#Compose 混合
trans_resize_2=transforms.Resize(512)
trans_compose=transforms.Compose([trans_resize_2,trans_totensor])
img_resize_2=trans_compose(img)
writer.add_image("Resize",img_resize_2,3)

#RandomCrop 随机裁剪
trans_random=transforms.RandomCrop(256)  #hw->(512,1025)
trans_compose_2=transforms.Compose([trans_random,trans_totensor])
for i in range(10):
    img_crop=trans_compose_2(img)
    writer.add_image("RandomCrop256",img_crop,i)


writer.close()