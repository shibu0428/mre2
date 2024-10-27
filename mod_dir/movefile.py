import glob
import os



def movefiles():
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
    out_folder="0710"
    in_path = '*_121DE.csv'
    folder_path='../dataset/rawcsv/'+out_folder+'/'


    flist = glob.glob(in_path)
    print(flist)


if __name__ == "__main__":
    movefiles()