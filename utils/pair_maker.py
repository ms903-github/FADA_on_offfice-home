import torch
import pandas as pd
import glob
import itertools
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import argparse

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_per_class",
        type=int,
        default=3
    )
    parser.add_argument(
        "--class_num",
        type=int,
        default=45
    )
    parser.add_argument(
        "--class_init",
        type=int,
        default=10
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="csv45_2"
    )

    return parser.parse_args()

def main():
    args = get_arguments()
    source_path = []
    target_path = []
    s_data = pd.read_csv(os.path.join(args.csv_path, "Product_train.csv"))
    s_pathlist = []
    s_labellist = []
    for _, row in s_data.iterrows():
        path, label = row["image_path"], row["label"]
        #if int(label) >= class_num:
        #    break
        s_pathlist.append(path)
        s_labellist.append(label)    
    for i in range(args.class_init, args.class_init+args.class_num):
        class_list = []
        for j in range(len(s_pathlist)):
            if int(s_labellist[j]) == i:
                class_list.append(s_pathlist[j])
        source_path.append(class_list)     #(source/target)_pathのi番目にラベルiのパスの集団が入っている
    t_data = pd.read_csv(os.path.join(args.csv_path, "RealWorld_few.csv"))
    t_pathlist = []
    t_labellist = []
    for _, row in t_data.iterrows():
        path, label = row["image_path"], row["label"]
        t_pathlist.append(path)
        t_labellist.append(label)
    for i in range(args.class_init, args.class_init+args.class_num):
        class_list = []
        for j in range(len(t_pathlist)):
            if int(t_labellist[j]) == i:
                class_list.append(t_pathlist[j])
        target_path.append(class_list)     #(source/target)_pathのi番目にラベルiのパスの集団が入っている
    G1_pairs = []   #ソース・ソース（同ラベル）
    G2_pairs = []   #ソース・ターゲット(同ラベル)
    G3_pairs = []   #ソース・ソース(異ラベル)
    G4_pairs = []   #ソース・ターゲット(異ラベル)
    G_pairs = []
    for i in range(args.class_num):
        for x, y in itertools.combinations(source_path[i], 2):
            G1_pairs.append([x, y, 0])

    print("G1")
    for i, k in itertools.combinations(range(args.class_num), 2):
        for x, y in itertools.product(source_path[i], source_path[k]):
            G3_pairs.append([x, y, 2])
            
    print("G3")
    for i in range(args.class_num):
        for k in range(args.class_num):
            for x, y in itertools.product(source_path[i], target_path[k]):
                if i == k:
                    G2_pairs.append([x, y, 1])
                else:
                    G4_pairs.append([x, y, 3])


    df1 = pd.DataFrame(G1_pairs)
    df2 = pd.DataFrame(G2_pairs)
    df3 = pd.DataFrame(G3_pairs)
    df4 = pd.DataFrame(G4_pairs)

    df1.to_csv(os.path.join(args.csv_path, "G1.csv"))
    df2.to_csv(os.path.join(args.csv_path, "G2.csv"))
    df3.to_csv(os.path.join(args.csv_path, "G3.csv"))
    df4.to_csv(os.path.join(args.csv_path, "G4.csv"))

if __name__ == "__main__":
    main()