#学習に用いるパラメーターを設定する
#学習に用いるデータの種類・ファイル名を決定する

#0505現在　データは　
#27*7　150フレーム で構成
#最親ディレクトリMREの下にdatasetがありそこに
#{モーション名}/{モーション名}N.txt
#N=ファイル番号　0からスタート



#ここにモーション名ライブラリを作成 
#fp="../dataset/",
fp="../dataset/0617/"
motions={
    0:"guruguru_stand",
    1:"suburi",
    2:"udehuri",
    3:"iai",
    4:"sit_stop",
    5:"sit_udehuri",
    6:"stand_nautral",
    7:"scwat",
    8:"fencing_stand",
}
'''
motions={
    0:"suburi",
    1:"iai",
    2:"stand_neutral",
}
'''

#学習パラメータ
learn_par={
    "Lnum_s":0, #学習の添え字スタート　この値を含む添え字から
    "Lnum_e":17,#この値の添え字(含まない)までを読み込み
    "Tnum_s":17,#テストの添え字スタート　この値を含む添え字から
    "Tnum_e":20,#この値の添え字(含まない)までを読み込み
    "fra_s":0,  #使用するフレームのスタート
    "fra_e":140, #使用するフレームのエンド
    "fra_seq":20,#セクションのフレーム数
}

#学習モデルを保存する=1
model_save=1
save_name='model/''model2'+'.pth'

#parts_cutを行うときのオプション
#nnflat_parts.pyを実行
parts_option=0
'''
0:"full_body"
1:"upper_body"
2:"lower_body"
3:"left_arm"
4:"right_arm"
5:"left_leg"
6:"right_leg"
'''