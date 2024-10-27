import numpy as np

import sys
sys.path.append('..')

from lib import readfile_new26 as rf
from lib import partsset

#ファイルパス=..../0430/suburi/
#return dataset[読んだファイルの数][frame][27parts][4dof]
def dataload_1motion(fpath,set_num_start,set_num_end,frame_s,frame_e):
    f_num=set_num_end-set_num_start
    frame=frame_e-frame_s
    dataset=np.empty((f_num,frame,26,4))
    #print(dataset.shape)
    for k in range(f_num):
        k_num=k+set_num_start
        fp=fpath+str(k_num)+".txt"
        qdata,xdata=rf.file_sep(fp)
        #print(data)
        dataset[k]=qdata[frame_s:frame_e]
        #print(fp,"を読み込みました")
    return dataset

#ファイルパス=..../0430/suburi/suburi
#return dataset[読んだファイルの数][frame][27parts][4dof]
def dataload_frame_seq(fpath,set_num_start,set_num_end,start_frame,end_frame,frame_sep):
    frames=end_frame-start_frame
    f_num=set_num_end-set_num_start
    n=int(frames/frame_sep)
    #dataset=np.empty((f_num*n,int(frames/n),27,4))
    dataset=np.empty((f_num*n,frame_sep,26,4))
    for i in range(n):
        sf=start_frame+i*frame_sep
        dataset[i*f_num:(i+1)*f_num]=dataload_1motion(fpath,set_num_start,set_num_end,sf,sf+frame_sep,)
    return dataset


#ファイルパス="../dataset/0430/"
#return dataset[種類*数*分割][frames/分割][27][4]
def dataloading(fpath,dict,set_num_start,set_num_end,start_frame,end_frame,frame_sep):
    frames=end_frame-start_frame
    f_num=set_num_end-set_num_start
    n=int(frames/frame_sep)*f_num
    dataset=np.empty((len(dict)*n,frame_sep,26,4))
    print(len(dict))
    for i in range(len(dict)):
        fp=fpath+dict.get(i)+"/"+dict.get(i)
        dataset[i*n:i*n+n]=dataload_frame_seq(fp,set_num_start,set_num_end,start_frame,end_frame,frame_sep)
    return dataset



#debug用
if __name__ == "__main__":
    dict_sample={
        0:"walk",
        1:"suburi",
        2:"udehuri"
    }
    fp_sample="../dataset/0430/"
    set_num_start_sample=5
    set_num_end_sample=10
    #frames_sample=10

    #dataset=dataload(fp_sample,dict_sample,set_num_start_sample,set_num_end_sample,0,10)
    #dataset=dataload_frame_seq(fp_sample,set_num_start_sample,set_num_end_sample,0,50,10)
    dataset=dataloading(fp_sample,dict_sample,set_num_start_sample,set_num_end_sample,0,50,10)
    print(dataset.shape)