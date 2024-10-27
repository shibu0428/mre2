#接続、保存、学習、試験を一括(consolidation)で行う
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#必要モジュールのimport
import socket
import time
import struct
import numpy as np
import winsound
import os

#torch関連の読み込み
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
import torchsummary

#自作関数読み込み
from lib import readfile as rf
from lib import partsset
from learning.archive_nn import dataload as dl
#partsのセットを行う
from lib import partsset as ps

fp1="C:/Users/tomok/mre/renshuu0.txt"
fp2="C:/Users/tomok/mre/dataset/before/iai/iai0.txt"
wxyz=rf.xyzq4(fp1)
print(wxyz.shape)
a=wxyz[0]

# 4種類の配列と3種類の配列を保持するためのリスト
four_item_lists = []
three_item_lists = []

# データを7つずつ取り出して分割
for i in range(0, len(a), 7):
    four_item_lists.append(a[i:i+4])
    three_item_lists.append(a[i+4:i+7])

# リストをNumPy配列に変換
four_item_arrays = np.array(four_item_lists)
three_item_arrays = np.array(three_item_lists)
t2=three_item_arrays
# 結果を表示
#print("4種類の配列: ", four_item_arrays)
print("3種類の配列: ", three_item_arrays)
xyz_coordinates = three_item_arrays.reshape(-1, 3)[:27]

# 3次元プロット
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(xyz_coordinates[:, 0], xyz_coordinates[:, 1], xyz_coordinates[:, 2], marker='o',color='red')

wxyz=rf.xyzq4(fp2)
print(wxyz.shape)
a=wxyz[0]

# 4種類の配列と3種類の配列を保持するためのリスト
four_item_lists = []
three_item_lists = []

# データを7つずつ取り出して分割
for i in range(0, len(a), 7):
    four_item_lists.append(a[i:i+4])
    three_item_lists.append(a[i+4:i+7])

# リストをNumPy配列に変換
four_item_arrays = np.array(four_item_lists)
three_item_arrays = np.array(three_item_lists)
print("3種類の配列: ", three_item_arrays)
ax.plot(xyz_coordinates[:, 0], xyz_coordinates[:, 1], xyz_coordinates[:, 2], marker='o',color='blue')

print("差: ", three_item_arrays[:26]-t2)

# ラベルの設定
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

# プロットを表示
plt.show()