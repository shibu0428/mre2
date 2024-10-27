import glob
import os

def rename_csv():
    path = '*.csv'
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
        "0D7A2",
        "0FC42",
        "12AA1",
        "1437E",
        "121DE",
        "13D54",
    ]
    # txtファイルを取得する
    flist = glob.glob(path)