import socket
import time
import struct
import binascii
import winsound


dof_parts=7 #つかうデータ=0,1,2,3==クォータニオン->dof=4
            #4,5,6=position->dof=7

host = ''
port = 5002

outfile=input("output file name?")

udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
udp_socket.bind((host, port))

# 受信用のバッファーサイズを設定
buffer_size = 8192

print(f"UDP 受信開始。ホスト: {host}, ポート: {port}")
max_sec=3
max_frame=max_sec*30
sum=0
flag=0
n=0
n_max=15
flames=1
ijou_time=time.time()
last_time = time.time()
time.sleep(1.5)
winsound.Beep(1400, 500)

while True:
    try:
        # データを受信
        data, addr = udp_socket.recvfrom(buffer_size)
        #if(len(data)>1600):raise Exception(f)
        with open(outfile+str(n)+'.txt', mode='a') as f:
            #data2=data.split(b'tran')
            #data3=data2.split(b'')
            f.write(str(data))
            f.write(f"\n")
            f.close()
        exit()

        
    except OSError as e:
        # エラーが発生した場合は表示
        print(f"エラー: {e}")
