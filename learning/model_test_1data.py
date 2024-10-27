# 準備あれこれ
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn
seaborn.set()
from torch.utils.data import Dataset
import sys

# PyTorch 関係のほげ
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from torchvision.datasets import FashionMNIST
import torchsummary
import csv
import torch.nn.functional as F

#自作関数軍
import sys
sys.path.append('..')
from lib import readfile as rf
#partsのセットを行う
from lib import partsset as ps
#データをファイルから読み込むためのローダ
import learning.archive_nn.dataload as dl


#---------------------------------------------------
#パラメータここから
# クラス番号とクラス名
dataset_path="0710"

motions=[
    "guruguru_stand",
    "suburi",
    "udehuri",
    "iai",
    "sit_stop",
    "sit_udehuri",
    "stand_nautral",
    "scwat",
    "fencing_stand",
]
args = sys.argv
choice_num=int(args[1])
choice_motion=motions[choice_num]

dev=[
    #"0D7A2",
    #"0FC42",
    "12AA1",
    "1437E",
    #"121DE",
    #"13D54"
]



model_save=0        #モデルを保存するかどうか 1なら保存
data_frames=20       #学習1dataあたりのフレーム数
all_data_frames=2000#元データの読み取る最大フレーム数

choice_mode=1   #テストのチョイスを変更する
fc1=1024
fc2=2048
model_path="./model/0825_model_dev2_20_sp.path"
#パラメータここまで
#----------------------------------------------------------------------------------

data_cols=7*len(dev)       #1dataの1フレームのデータ数


#cudaの準備
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#print(device)
#print(torch.cuda.is_available())


#データロード開始
#print("data load now!")
np_dataset=np.zeros((all_data_frames,data_cols))
np_parts=np.zeros((all_data_frames,7))
frame_check=0
data_check=0

for j in range(len(dev)):
    flag=0
    with open('../dataset/'+dataset_path+'/'+choice_motion+'_'+dev[j]+'.csv') as f:
        reader = csv.reader(f)
        for row in reader:
            if flag<150:
                flag+=1
                continue
            np_parts[frame_check]=row[1:]#
            frame_check+=1
            if frame_check==all_data_frames:
                frame_check=0
                np_dataset[:,7*j:7*(j+1)]=np_parts
                break
        #print(np_parts)
#print(np_dataset.shape)


class MLP4(nn.Module):

    # コンストラクタ． D: 入力次元数， H1, H2: 隠れ層ニューロン数， K: クラス数
    def __init__(self, D, H1, H2,K):
        super(MLP4, self).__init__()
        # 4次元テンソルで与えられる入力を2次元にする変換
        self.flatten = nn.Flatten()
        # 入力 => 隠れ層1
        self.fc1 = nn.Sequential(
            nn.Linear(D, H1), nn.Sigmoid()
        )
        # 隠れ層1から隠れ層2へ
        self.fc2 = nn.Sequential(
            nn.Linear(H1,H2), nn.Sigmoid()
        )
        # 隠れ層 => 出力層
        self.fc3 = nn.Linear(H2, K) # 出力層には活性化関数を指定しない
        # モデルの出力を計算するメソッド
    def forward(self, X):
        X = self.flatten(X)
        X = self.fc1(X)
        X = self.fc2(X)
        X = self.fc3(X)
        return X


# ネットワークモデル

net = MLP4(data_frames*data_cols,fc1,fc2, len(motions)).to(device)
net.load_state_dict(torch.load(model_path))

# 損失関数（交差エントロピー）
loss_func = nn.CrossEntropyLoss(reduction='sum')

np_1data=np.zeros((data_frames,data_cols))
for i in range(all_data_frames-21):
    np_1data=np_dataset[i:i+20,:].reshape(-1, data_frames * data_cols)
    t_data = torch.from_numpy(np_1data).to(device)
    X = t_data.float()  # 入力データをFloat型に変換
    Y = net(X)           # 一つのバッチ X を入力して出力 Y を計算
    choice_num_tensor = torch.tensor([choice_num] * X.size(0), dtype=torch.long, device=device)  # 正解ラベルをテンソルに変換
    loss = loss_func(Y, choice_num_tensor)  # 正解ラベル lab に対する loss を計算
    loss_value=loss.item()
    probabilities = F.softmax(Y, dim=1)
    max_prob, predicted_label = torch.max(probabilities, dim=1)
    print(Y.argmax(dim=1).item(),",\t",f"{loss_value:.6f}",',\t',f"{max_prob.item():.6f} ,")
