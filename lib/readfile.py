#27*7dataファイル(パス＆ファイル名)からデータをnumpy配列にして返す
import numpy as np


#パスに入ったデータからすべての要素をnumpy配列に格納
def  readfile2np(fp):
    data_list = []
    print(fp)
    with open(fp, 'r') as file:
        for line in file:
            # 行からスペースまたはタブで区切られたデータを取得し、floatに変換してリストに追加
            data_list.append(list(map(float, line.strip().split())))

    # NumPyのarrayに変換
    data_2d_array = np.array(data_list)

    return data_2d_array

#data[frame数][189:7dof_data*27]の配列をdata[frame][parts][7dof]の形式に変更
def parser_q4xyz(data):
    if data.size == len(data) * 27 * 7:
        reshaped_data = data.reshape(len(data), 27, 7)
        #print("parse: ",reshaped_data.shape)
        return reshaped_data
    else:
        
        print(data.size)
        #4dofならそのまま吐き出す
        if data.size == len(data)*27*4:
            print("4dofデータと認識")
            reshaped_data = data.reshape(len(data), 27, 4)
            return reshaped_data
        print("エラー: 元のデータの要素数と新しい形状の要素数が一致しません。")
        exit()

#parseされた7dofdataをqxyzの配列とxyzの配列のふたつに分ける
def separate(dofdata):
    qdofdata=dofdata[0:len(dofdata),0:len(dofdata[0]),0:4]
    #qdofdata=dofdata[0:181][0:27][0:4]
    xyzdofdata=dofdata[0:len(dofdata),0:len(dofdata[0]),4:7]
    return qdofdata,xyzdofdata


#fpからセパレートまでを一括で行う
def file_sep(fp):
    return separate(parser_q4xyz(readfile2np(fp)))

def xyzq4(fp):
    return readfile2np(fp)




#debug用
if __name__ == "__main__":
    '''
    data=readfile2np('sample_int.txt')
    print("1frameのデータ数",len(data[0]))
    print('data数',len(data))
    parse_data=parser_q4xyz(data)
    q,xyz=separate(parse_data)
    print(q.astype(int))
    print(xyz.shape)
    '''
    fp='../dataset/sample/sample_int.txt'
    q,xyz=file_sep(fp)
    print(q.shape)