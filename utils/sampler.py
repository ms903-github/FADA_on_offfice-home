import glob
import os
import random
import shutil
import pandas as pd
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
    df = pd.read_csv(os.path.join(args.csv_path, "RealWorld_train.csv"))
    sample = pd.DataFrame(columns=["image_path", "label"])
    for clss in range(args.class_init, args.class_init+args.class_num):
        cand = pd.DataFrame(columns=["image_path", "label"])
        for _, row in df.iterrows():
            path, label = row["image_path"], row["label"]
            if int(label) == clss:
                cand = cand.append(row)
        for i in range(args.data_per_class):
            idx = random.randint(0, len(cand)-1)
            sample = sample.append(cand.iloc[idx])
    sample.to_csv(os.path.join(args.csv_path, "RealWorld_few.csv"))

if __name__ == "__main__":
    main()
