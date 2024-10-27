import socket
import struct

# UDPの受信設定
udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
udp_socket.bind(("192.168.10.110", 5002))  # IPアドレスを指定してバインド

filename=input("ファイル名")
print("Waiting for UDP data")
with open(filename+'.csv',mode='a') as f:
    while True:
        data, addr = udp_socket.recvfrom(1024)  # データを受信
        for i in range(6):
            f.write(str(struct.unpack('<ffffffff', data[i*40:i*40+32])).replace(")(", ",").replace("(", "").replace(")", ""))
            f.write(",")
            f.write(str(struct.unpack('<q', data[i*40+32:i*40+40])).replace(")(", ",").replace("(", "").replace(")", ""))
            #print(struct.unpack('<q', data[i*40+32:i*40+40]))
        f.write(f"\n")

'''
print("Waiting for UDP data")
with open(filename+'.csv',mode='a') as f:
    while True:
        data, addr = udp_socket.recvfrom(1024)  # データを受信
        for i in range((int(len(data)/4))):
            if i%9 == 8:
                f.write(str(struct.unpack('<f', data[i*4:i*4+4])).replace(")(", ",").replace("(", "").replace(")", ""))
            else:
                f.write(str(struct.unpack('<f', data[i*4:i*4+4])).replace(")(", ",").replace("(", "").replace(")", ""))
        f.write(f"\n")
'''        
        
