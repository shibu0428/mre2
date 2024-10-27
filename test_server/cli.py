#mocopiの送信側を偽装するクライアントプログラム

import socket
import time
HOST='localhost'
PORT=42842

frames=0
fp='rawdata/damy.dat'

data=[]

# バイナリファイルを読み込みモードでオープンする
with open(fp, 'rb') as f:
    for one_data in f:
        one_data=one_data.replace(b'\n',b'')
        data.append(one_data)
print(data[0])
print(len(data))

with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s: 
    s.connect((HOST, PORT)) 
    
    while True:
        #print(frames)
        s.sendall(data[frames])
        frames+=1
        
        if frames>900:
            frames=0
        time.sleep(0.033)

