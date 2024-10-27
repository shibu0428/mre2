#mocopiの生データを受け取り28*7パーツ分をフラットなリストにして返す
import time
import struct


def parse(data):
    datalist=[]
    bnid_list = data.split(b'bnid')[1:]
    for id,part in enumerate(bnid_list):
        tran_btdt_data = part.split(b'tran')[1:]
        dofdata = tran_btdt_data[0].split(b'btdt')[0]
        if len(dofdata) > 28:
            dofdata=dofdata[:28]
        for id,i in enumerate(range(0, 28, 4)):
            float_value = struct.unpack('<f', dofdata[i:i+4])
            datalist.append(float_value)
    return datalist

