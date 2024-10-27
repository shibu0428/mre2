import glob
import os

# 拡張子.txtのファイルを取得する


def partial_match_index(lst, word):
    for i, item in enumerate(lst):
        if item in word:
            return i
    return -1  # 一致するものがなければ-1を返す

#backup_csv("モーション名","デバイス名","元ファイル名")
def backup_csv(motion,dev,path):
    # txtファイルを取得する
    
    list = glob.glob(path)
    print('変更完了')

if __name__ == "__main__":
    backup_csv("walk","SAMPLE","sample.")