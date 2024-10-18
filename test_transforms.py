from PIL import Image
from torchvision import transforms

img_path="dataset/train/ants_image/0013035.jpg"
img=Image.open(img_path)
print(type(img))

tensor_trans=transforms.ToTensor() #ToTensor是一个类
tensor_img =tensor_trans(img) #把img转成tensor类型