import pandas as pd
import torch

class_n = 65
def get_class_num(train_csv_file='./csv/Product_train.csv', n_classes=class_n):
    """ get the number of samples in each class """

    df = pd.read_csv(train_csv_file)
    nums = {}
    for i in range(n_classes):
        nums[i] = 0
    for i in range(len(df)):
        nums[df["label"][i]] += 1
    class_num = []
    for val in nums.values():
        class_num.append(val)
    class_num = torch.tensor(class_num)

    return class_num


def get_DCD_weight(file1="./csv/G1.csv", file2="./csv/G2.csv", file3="./csv/G3.csv", file4="./csv/G4.csv"):
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    df3 = pd.read_csv(file3)
    df4 = pd.read_csv(file4)
    class_num = [0, 0, 0, 0]
    class_num[0] = len(df1)
    class_num[1] = len(df2)
    class_num[2] = len(df3)
    class_num[3] = len(df4)
    class_num = torch.Tensor(class_num)

    total = class_num.sum().item()
    frequency = class_num.float() / total
    median = torch.median(frequency)
    class_weight = median / frequency

    return class_weight


def get_class_weight(train_csv_file='./csv/Product_train.csv', n_classes=class_n):
    """
    Class weight for CrossEntropy in Flowers Recognition Dataset
    Class weight is calculated in the way described in:
        D. Eigen and R. Fergus, “Predicting depth, surface normals and semantic labels with a common multi-scale convolutional architecture,” in ICCV,
        openaccess: https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Eigen_Predicting_Depth_Surface_ICCV_2015_paper.pdf
    """

    class_num = get_class_num(train_csv_file, n_classes)
    total = class_num.sum().item()
    frequency = class_num.float() / total
    median = torch.median(frequency)
    class_weight = median / frequency

    return class_weight