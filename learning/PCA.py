#主成分分析をしたい

####################################
#事前準備
####################################
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn
seaborn.set()

#自作関数軍
import sys
sys.path.append('..')
from lib import readfile as rf
#partsのセットを行う
from lib import partsset as ps
#データをファイルから読み込むためのローダ
import learning.archive_nn.dataload as dl
#motionの種類や使用するファイル数などのパラメータ
import learning.archive_nn.param as par

#ファイルパス、種類クラスの親まで,/入り
fp=par.fp
# クラス番号とクラス名
#label_map.keys()
#"suburi"=label_map.get(1)
labels_map = par.motions

#データ読み込み
#ファイル添え字の設定
Lnum_s=par.learn_par["Lnum_s"]
Lnum_e=par.learn_par["Lnum_e"]
Lnum=Lnum_e-Lnum_s

#frameの設定
fra_s=par.learn_par["fra_s"]
fra_e=par.learn_par["fra_e"]
fra_sep=par.learn_par["fra_seq"]


#データロード開始
print("data load now!")
np_data = dl.dataloading(fp,labels_map,Lnum_s,Lnum_e,fra_s,fra_e,fra_sep)

####################################
#事前準備完了   np_dataに[data数][フレーム数][パーツ数][DoF]が入っている
#添え字についてはparam.py参照
####################################

#np_dataを[data数][フラットな配列]にする
n=Lnum_e-Lnum_s

print(np_data.shape)
flat_data=np.reshape(np_data,(len(np_data),len(np_data[0])*27*4))
#print(flat_data.shape)

# 平均
Xm = np.mean(flat_data, axis=0)
print('平均:', Xm)

# 平均を引いたあとのデータ配列Xを作成
X = flat_data - Xm
N, D = X.shape
print('X.shape:', X.shape)
# 分散共分散行列の固有値と固有ベクトルを求める（Xの特異値分解経由で）
_, sval, Vt = np.linalg.svd(X, full_matrices=False)
eval = sval**2/N
U = Vt
'''
for d in range(D):
    print(f'{d+1}番目の固有値:{eval[d]:.2f}   固有ベクトル:', U[d, :])
    if d>88:
        break
'''
W = U[:3, :]
print(W.shape)
print(W)
Y = X @ W.T # y = Wx の計算
print(Y.shape)
print(Y[:, :]) # 最初の5人分を表示


motions={
    0:"walk",
    1:"suburi",
    2:"udehuri",
    3:"iai",
    4:"sit_stop",
    5:"sit_udehuri",
    6:"stand_neutral",
    7:"scwat",
    8:"fencing_stand",
}

color_list={
    0:"blue",
    1:"orange",
    2:"green",
    3:"red",
    4:"purple",
    5:"brown",
    6:"pink",
    7:"olive",
    8:"cyan",
}

# 3Dプロットの作成
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
fnum=int(len(np_data)/len(motions))
for i in range(len(motions)):
    color=color_list[i]
    for j in range(fnum):
        #print(i,j,fnum*i+j)
        ax.scatter(Y[fnum*i+j, 0], Y[fnum*i+j, 1],Y[fnum*i+j,2],c=color)
#ax.scatter(Y[nList, 0], Y[nList, 1])
ax.axvline(0, linestyle='-', color='gray')
ax.axhline(0, linestyle='-', color='gray')
ax.set_xlim(-8, 8)
ax.set_ylim(-8, 8)
ax.set_aspect('equal')
#for n in nList:
    #plt.annotate(f'{n}', (Y[n, 0]+2, Y[n, 1]+2))
plt.show()

# 寄与率と累積寄与率
c = eval / np.sum(eval)
print()
print('寄与率:　　', c)
cc = np.cumsum(c)  # np.cumsum がやってるのは以下と同じ
'''
cc = np.empty_like(c)
cc[0] = c[0]
for d in range(1, len(c)):
    cc[d] = cc[d-1] + c[d]
'''
print('累積寄与率:', cc)
#print(fnum)