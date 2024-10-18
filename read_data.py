from torch.utils.data import Dataset
from PIL import Image
import os
class MyData(Dataset):

    def __init__(self,root_dir,label_dir):
        self.root_dir=root_dir #训练集 train
        self.label_dir=label_dir #训练集子类 ants，beef
        self.path=os.path.join(self.root_dir,self.label_dir) #join 联立路径
        self.img_path=os.listdir(self.path) #构成list，找到图片集合

    def __getitem__(self, idx):
        img_name=self.img_path[idx] #图片名称
        img_item_path=os.path.join(self.root_dir,self.label_dir,img_name) #图片的相对路径
        img=Image.open(img_item_path) #img
        label=self.label_dir  #标签
        return img,label  #返回图片属性和标签

    def __len__(self):
        return len(self.img_path) #img个数

root_dir="dataset/train"
ants_label_dir="ants_image"
ants_dataset=MyData(root_dir,ants_label_dir)

bees_label_dir="bees_image"
bees_dataset=MyData(root_dir,bees_label_dir)

train_dataset=ants_dataset+bees_dataset #拼接数据集