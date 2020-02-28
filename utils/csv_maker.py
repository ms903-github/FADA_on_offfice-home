import glob
import os
import random
import shutil
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="./dataset/RealWorld"    #ルートで実行することを想定したパス指定
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./csv55-2"
    )
    parser.add_argument(
        "--class_num",
        type=int,
        default=55
    )
    parser.add_argument(
        "--domain_name",
        type=str,
        default="RealWorld"
    )
    parser.add_argument(
        "--class_init",
        type=int,
        default=10
    )

    return parser.parse_args()

def main():
    args = get_arguments()
    labeldict = {}
    img_paths = []
    img_labels = []
    f = open("./dataset/label-code.txt")
    lines = [tmp.strip() for tmp in f.readlines()]
    f.close()
    for i in range(args.class_init, args.class_init+args.class_num):
        label, code = lines[i].split()
        labeldict.setdefault(label, int(code))   #フォルダがクラス名ではなくクラスコードになっていればlabeldictは不要
    pathlist = glob.glob(args.dataset_dir + "/*/*.jpg")
    for path in pathlist:
        tmp = path.split("/")
        clss = tmp[3]                      #パスを"/"で区切ったとき、3番目にクラス名が来ているため
        
        if clss in labeldict:
            img_labels.append(labeldict[clss])
            img_paths.append(path)
            
    
    result_df = pd.DataFrame({
        "image_path" : img_paths,
        "label" : img_labels

    })
    train_df, test_df = train_test_split(result_df, train_size=0.3)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    train_df.to_csv(os.path.join(args.save_dir, str(args.domain_name+"_train.csv")), index=None, mode="w")
    test_df.to_csv(os.path.join(args.save_dir, str(args.domain_name+"_test.csv")), index=None, mode="w")

if __name__ == "__main__":
    main()


