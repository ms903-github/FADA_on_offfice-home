import os
import glob
from torchvision import transforms
#from libs import *
from PIL import Image
from joblib import Parallel, delayed
import shutil

resize_tensor = transforms.Compose([
    transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
    #transforms.Normalize(mean=get_mean(), std=get_std())
])
convert_per_pict = 10

if os.path.exists("dataset/converted_RealWorld"):
    exit()
pathlist = glob.glob("dataset/*/*/*.jpg")
shutil.copytree("dataset/RealWorld", "dataset/converted_RealWorld")
shutil.copytree("dataset/Product", "dataset/converted_Product")

def convert(path):
    for i in range(convert_per_pict):
        img = Image.open(path)
        img_t = resize_tensor(img)
        newpath = path[:8]+"converted_" + path[8:-4] + "_" + str(i) + path[-4:]
        img_t.save(newpath)
        
Parallel(n_jobs=-1)([delayed(convert)(path) for path in pathlist])

# for path in pathlist:
#     img = Image.open(path)
#     for i in range(convert_per_pict):
#         img_t = resize_tensor(img)
#         newpath = path[:8]+"converted_" + path[8:-4] + "_" + str(i) + path[-4:]
#         img_t.save(newpath)
        