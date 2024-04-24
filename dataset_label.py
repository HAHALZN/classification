import os
import pandas as pd

data_name = ['APTOS2019', 'IDRiD','Messidor', 'DDR','Messidor-2', 'EyePACS']
def read_txt(txt_path):
    df = pd.read_csv(txt_path, delimiter=" ")
    for i in range(len(df)):
        df["path"][i] = df["path"][i].split("\\")[-1]
    print(df)
    save_dir = os.path.dirname(txt_path).replace("labels","new_labels")
    os.makedirs(save_dir, exist_ok=True)
    df.to_csv(os.path.join(save_dir, os.path.basename(txt_path)), index=False, header=['path','label'], sep=" ")
    

def modify_label(path):
    #读取path中的label文件
    label_file = os.path.join(path, "labels")
    train_file = os.path.join(label_file, "train.txt")
    test_file = os.path.join(label_file, "test.txt") 
    read_txt(train_file)    
    read_txt(test_file)    

def main():
    # 获取不同数据集的label
    path = "/root/autodl-fs/retinal/"
    datasets = os.listdir(path)
    for dataset in datasets:
        dataset_path = os.path.join(path, dataset)
        if os.path.isdir(dataset_path):
            if dataset in data_name:
                print(f"Processing {dataset} dataset...")
                # 读取label
                modify_label(dataset_path)
            else:
                print(f"Skipping {dataset} dataset...")


if __name__ == '__main__':
    main()