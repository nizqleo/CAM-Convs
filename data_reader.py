import os
from PIL import  Image
import numpy as np

import torch
import torchvision
from torch.utils import data
from torchvision import transforms 

class Sematics2D3D(data.Dataset):
    def __init__(self, root, list, transforms=None):
        self.transforms=transforms
        self.list = list

        self.imgs_rgb=[]
        self.imgs_normals=[]
        self.imgs_depth=[]

        for number in self.list:
            area_dir=os.path.join(root, 'area_'+ str(number)+'/data') 
            data_name = os.listdir(os.path.join(area_dir, 'rgb'))
            self.imgs_rgb += [os.path.join(area_dir,'rgb',img) for img in data_name]
            self.imgs_normals += [os.path.join(area_dir,'normals', img.replace('rgb', 'normals')) for img in data_name]
            self.imgs_depth += [os.path.join(area_dir,'depth', img.replace('rgb', 'depth')) for img in data_name]

    def __getitem__(self, index):
        rgb_path = self.imgs_rgb[index]
        depth_path = self.imgs_depth[index]
        normals_path = self.imgs_normals[index]
       
        rgb = Image.open(rgb_path)
        depth = Image.open(depth_path)
        normals = Image.open(normals_path)
        
        if self.transforms:
            rgb = self.transforms(rgb)
            depth = self.transforms(depth)
            normals = self.transforms(normals)
            
        
        rgb = torch.from_numpy(np.asarray(rgb)).type(torch.FloatTensor)
        depth = torch.from_numpy(np.asarray(depth)).type(torch.FloatTensor)
        normals = torch.from_numpy(np.asarray(normals)).type(torch.FloatTensor)
        

        return rgb, depth, normals
    
    def __len__(self):
        return len(self.imgs_rgb)