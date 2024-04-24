
import os

import pandas as pd

target_data = ["APTOS2019"]
source_data = ["EyePACS(Kaggle)","IDRiD","DDR","Messidor","Messidor-2"]

def data_process(file_path):
    new_path = os.path.join(file_path, "labels")
    
    if os.path.isdir(new_path):
        # 获取目录中的所有文件和子目录
        files = os.listdir(new_path)
        # 遍历所有文件和子目录
        for file in files:
            # 构建文件的完整路径
            file_path = os.path.join(new_path, file)
            if "train.txt" in file_path:
                df = pd.read_csv(file_path, sep=" ")
                print("训练集共计{}个样本".format(len(df)))
                labels = df['label'].unique()
                print("训练集共计{}个类别".format(labels))

            elif "test.txt" in file_path:
                df = pd.read_csv(file_path, sep=" ")
                print("测试集共计{}个样本".format(len(df)))
                labels = df['label'].unique()
                print("测试集共计{}个类别".format(labels))
    
            elif "val.txt" in file_path:
                df = pd.read_csv(file_path, sep=" ")
                print("验证集共计{}个样本".format(len(df)))
                labels = df["label"].unique()
                print("验证集共计{}个类别".format(labels))    

def stat(data_path):
    if os.path.isdir(data_path):
        # 获取目录中的所有文件和子目录
        files = os.listdir(data_path)
        # 遍历所有文件和子目录
        for file in files:
            # 构建文件的完整路径
            file_path = os.path.join(data_path, file)
            if file in target_data:
                print("--目标域数据统计--:{}".format(file))
                _ = data_process(file_path)
            elif file in source_data:
                print("--源域数据统计--{}".format(file))
                _ = data_process(file_path)
    else:
        raise ValueError("data_path is not a directory")
    



if __name__ == "__main__":
    path = "/root/autodl-fs/retinal"
    stat(path)