import torch
import numpy as np
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import glob
import os
import pandas as pd
import time
import random

def change_path(input_path, convert_per_pict=10):
    #change filepath for data augmentation
    i = random.randrange(convert_per_pict)
    output_path = input_path[:10] + "converted_" + input_path[10:-4] + "_" + str(i) + input_path[-4:]
    return output_path

def load_pair(csv1, csv2, csv3 = None, csv4 = None, transform = None):
    
    df1 = pd.read_csv(csv1, index_col = 0)
    df2 = pd.read_csv(csv2, index_col = 0)
    if csv3:
        df3 = pd.read_csv(csv3, index_col = 0)
    if csv4:
        df4 = pd.read_csv(csv4, index_col = 0)
    data_num = min(len(df1), len(df2))                #総データ数はdata_num x4
    df1_sample = df1.sample(n = data_num)
    df2_sample = df2.sample(n = data_num)
    if csv3:
        df3_sample = df3.sample(n = data_num)
    if csv4:
        df4_sample = df4.sample(n = data_num)
    if csv3 and csv4:
        df = pd.concat([df1_sample, df2_sample, df3_sample, df4_sample])
    elif csv3:
        df = pd.concat([df1_sample, df2_sample, df3_sample])
    else:
        df = pd.concat([df1_sample, df2_sample])

    class MyDataset(Dataset):
        def __init__(self, dataframe, transform = transform):
            self.df = dataframe
            self.transform = transform
        def __len__(self):
            return len(self.df)
        def __getitem__(self, idx):
            img1 = Image.open(change_path(self.df.iloc[idx, 0]))
            img2 = Image.open(change_path(self.df.iloc[idx, 1]))
            label = int(self.df.iloc[idx, 2])
            sample = {
                "image1" : img1,
                "image2" : img2,
                "label" : label
            }
            if self.transform:
                sample["image1"] = self.transform(sample["image1"])
                sample["image2"] = self.transform(sample["image2"])
            img1.close()
            img2.close()
            return sample
    
    dataset = MyDataset(df, transform = transform)
    return dataset
    
def load_pict(data_per_class, s_load_path, t_load_path, transform = None, batch_size = 64):
    class DatatoDataset(Dataset):
        def __init__(self, data, label):
            self.datas = data
            self.labels = label
        def __len__(self):
            return len(self.labels)
        def __getitem__(self, idx):
            sample = [self.datas[idx], self.labels[idx]]
            return sample
    df = pd.read_csv(s_load_path)
    
    s_pathlist = []
    s_labellist = []
    s_datalist = []
    for _, row in df.iterrows():
        path, label = row["image_path"], row["label"]
        s_pathlist.append(path)
        s_labellist.append(int(label))
    for path in s_pathlist:
        img = Image.open(path)
        if transform:
            img_t = transform(img)
            s_datalist.append(img_t)
        else:
            s_datalist.append(img)
        img.close()
    
    df = pd.read_csv(t_load_path)
    
    t_pathlist = []
    t_labellist = []
    t_datalist = []
    for _, row in df.iterrows():
        path, label = row["image_path"], row["label"]
        t_pathlist.append(path)
        t_labellist.append(int(label))
    for path in t_pathlist:
        img = Image.open(path)
        if transform:
            img_t = transform(img)
            t_datalist.append(img_t)
        else:
            t_datalist.append(img)
        img.close()
    
    source_dataset = DatatoDataset(s_datalist, s_labellist)
    target_dataset = DatatoDataset(t_datalist, t_labellist)
    return (source_dataset, target_dataset)

def load_pict2(load_path, transform=None, offset=0):
    class MyDataset(Dataset):
        def __init__(self, file_path, transform):
            self.df = pd.read_csv(file_path)
            self.transform = transform

        def __len__(self):
            return len(self.df)
        
        def __getitem__(self, idx):
            img_path = self.df.iloc[idx]["image_path"]
            label = self.df.iloc[idx]["label"] - offset
            img = Image.open(change_path(img_path))
            if self.transform:
                img = self.transform(img)
            return img, label
    dataset = MyDataset(load_path, transform=transform)

    return(dataset)
    
def concat_dataset(g_dataset, s_trainset, t_trainset):
    class catDataset(Dataset):
        def __init__(self, g_dataset, s_trainset, t_trainset, transform = None):
            self.ds1 = g_dataset
            self.ds2 = s_trainset
            self.ds3 = t_trainset
            self.transform = transform
        def __len__(self):
            return len(self.ds1)
        def __getitem__(self, idx):
            length = max(len(self.ds1), len(self.ds2), len(self.ds3))
            sample = {
                "image1" : self.ds1[idx % len(self.ds1)]["image1"],
                "image2" : self.ds1[idx % len(self.ds1)]["image2"],
                "label" : self.ds1[idx % len(self.ds1)]["label"],
                "s_traindata" : self.ds2[idx % len(self.ds2)][0],
                "s_trainlabel" : self.ds2[idx % len(self.ds2)][1],
                "t_traindata" : self.ds3[idx % len(self.ds3)][0],
                "t_trainlabel" : self.ds3[idx % len(self.ds3)][1]
            }
            if self.transform:
                sample["image1"] = self.transform(sample["image1"])
                sample["image2"] = self.transform(sample["image2"])
            
            return sample

    return catDataset(g_dataset, s_trainset, t_trainset)
