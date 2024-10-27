import glob
import os

# 拡張子.txtのファイルを取得する


def partial_match_index(lst, word):
    for i, item in enumerate(lst):
        if item in word:
            return i
    return -1  # 一致するものがなければ-1を返す

def rename_csv():
    path = '*'
    time=input("ファイルの名前の数字のみを入力")
    ext='.csv'
    path+=time+ext
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

    dev=[
        #"0D7A2",
        #"0FC42",
        "12AA1",
        "1437E",
        #"121DE",
        #"13D54",
    ]

    for i in range(len(motions)):
        print(str(i)+" = "+motions[i])
    n_motions=input("モーションを数字で選択")



    # txtファイルを取得する
    flist = glob.glob(path)
    # ファイル名を一括で変更する
    for file in flist:
        n_dev=partial_match_index(dev, file)
        if n_dev==-1:
            print("デバイス名エラー")
            continue
        os.rename(file, motions[int(n_motions)]+'_' + dev[int(n_dev)] + '.csv')

    list = glob.glob(path)
    print('変更完了')

if __name__ == "__main__":
    rename_csv()