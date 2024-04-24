# with open("/root/autodl-tmp/classification/datasets/retinal/DR/APTOS2019/train.txt","r")as f:
#     tmp=f.readlines()
#     cnt=0
#     with open("/root/autodl-tmp/classification/datasets/retinal/DR/APTOS2019/train_label.txt","w")as tl:
#         with open("/root/autodl-tmp/classification/datasets/retinal/DR/APTOS2019/train_unlabel.txt","w")as tu:
#             for i in tmp:
#                 if cnt%10 < 3:
#                     tl.write(i)
#                 else:
#                     tu.write(i)
#                 cnt+=1

import os

path="/root/autodl-tmp/classification/datasets/retinal/DR"

for i in os.listdir(path):
    # print(i)
    for j in os.listdir(os.path.join(path,i)):
        ret={"0":0,
             "1":0,
             "2":0,
             "3":0,
             "4":0}
        with open(os.path.join(path,i,j),"r")as f:
            tmp=f.readlines()
            for k in tmp:
                ret[k.split()[1].strip("\n")]+=1
            print(os.path.join(path,i,j))
            print(ret)



