using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Net.Sockets;
using System.Net;
using System;
using UnityEngine.Events;
using System.Timers;

/* UdpSystem.cs - UDPを用いたネットワーク通信プログラム -
 * 
 * Copyright(c) 2018 YounashiP
 * Released under the MIT license.
 * see https://opensource.org/licenses/MIT
 * version 0.1.1
 */

public class UDPSystem {

    /* Set : 自分と相手のIP、Port、Byte受け取り先関数を指定します
     * Receive : 受信を開始します
     * Send(byte[]): byteを送信します。　
     * Stop():　WebSocketを閉じます。
     */
    private class IPandPort
    {
        IPandPort(string ipAddr,int port)
        {
            this.ipAddr = ipAddr;
            this.port = port;
        }
        public string ipAddr;
        public int port;
        public UdpClient udpClient;
    }
    readonly int RETRY_SEND_TIME = 10; // ms

    static byte sendTaskCount = 0;
    static List<IPandPort> recList = new List<IPandPort>();

    bool finishFlag = false;
    bool onlyFlag = false;

    int sendHostPort = 6001;
    int sendHostPortRange = 0;

    Action<byte[]> callBack;


    public string recIP , sendIP ;
    public int recPort = 5000, sendPort = 5000;

    UdpClient udpClientSend;
    UdpClient tmpReceiver; //受信終了用TMP

    public UDPSystem(Action<byte[]> callback)
    {
        callBack = callback;
        
    }
    public UDPSystem(string rec_ip, int recport,  string send_ip, int sendport,Action<byte[]> callback, bool onlyflag = false) //オーバーロード 2
    {
        /* rec,send IP == null -> AnyIP */

        recIP = rec_ip;
        sendIP = send_ip;
        recPort = recport;
        sendPort = sendport;
        callBack = callback;
        onlyFlag = onlyflag;
    }

    public void Set(string rec_ip,int recport, string send_ip, int sendport, Action<byte[]> callback = null)
    {
        recIP = rec_ip;
        sendIP = send_ip;
        recPort = recport;
        sendPort = sendport;
        if (callback != null) callBack = callback;
    }
    public void SetSendHostPort(int port,int portRange = 0) //送信用 自己ポート設定
    {
        sendHostPort = port;
        sendHostPortRange = portRange;
    }

    int GetSendHostPort()
    {
        if (sendHostPortRange == 0) return sendHostPort;
        return UnityEngine.Random.Range(sendHostPort, sendHostPort + 1);
    }
    public void Finish() //エラー時チェック項目 : Close()が2度目ではないか
    {
        if (tmpReceiver != null) tmpReceiver.Close();
        else finishFlag = true;
    }
    public void Receive() // ポートの監視を始めます。
    {
        string targetIP = recIP; //受信
        int port = recPort;

        //if (recList.Contains(new IPandPort())) ;

        UdpClient udpClientReceive;

        if(targetIP == null) udpClientReceive = new UdpClient(new IPEndPoint(IPAddress.Any, port));
        else if (targetIP == "") udpClientReceive = new UdpClient(new IPEndPoint(IPAddress.Parse(ScanIPAddr.IP[0]), port));
        else udpClientReceive = new UdpClient(new IPEndPoint(IPAddress.Parse(targetIP), port));

        udpClientReceive.BeginReceive(UDPReceive, udpClientReceive);

        if (targetIP == null) Debug.Log("受信を開始しました。 Any " + IPAddress.Any + " " + port);
        else if (targetIP == "") Debug.Log("受信を開始しました。 Me " + ScanIPAddr.IP[0] + " " + port);
        else Debug.Log("受信を開始しました。" + IPAddress.Parse(targetIP) + " " + port);

        tmpReceiver = udpClientReceive;
    }
  
    void UDPReceive(IAsyncResult res) {// CallBack ポートに着信があると呼ばれます。

        if (finishFlag)
        {
            FinishUDP(res.AsyncState as UdpClient);
            return;
        }

        UdpClient getUdp = (UdpClient) res.AsyncState;
        IPEndPoint ipEnd = null;
        byte[] getByte;

        try
        { //受信成功時アクション
            getByte = getUdp.EndReceive(res, ref ipEnd);
            if(callBack!=null) callBack(getByte);
        }
        catch(SocketException ex)
        {
            Debug.Log("Error" + ex);
            return;
        }
        catch (ObjectDisposedException) // Finish : Socket Closed
        {
            Debug.Log("Socket Already Closed.");
            return;
        }

        if (finishFlag || onlyFlag) {
            FinishUDP(getUdp);
            return;
        }


        Debug.Log("Retry");
        getUdp.BeginReceive(UDPReceive, getUdp); // Retry
        
    }
    private void FinishUDP(UdpClient udp)
    {
        udp.Close();
    }

    public void Send_NonAsync(byte[] sendByte) //同期送信を行います。(未検証＆使用不要)
    {
        if(udpClientSend == null) udpClientSend = new UdpClient(new IPEndPoint(IPAddress.Parse(ScanIPAddr.IP[0]), GetSendHostPort()));
        udpClientSend.EnableBroadcast = true;

        try
        {
            udpClientSend.Send(sendByte, sendByte.Length,sendIP,sendPort);
        }
        catch (Exception e)
        {
            Debug.LogError(e.ToString());
        }
    }

    public void Send_NonAsync2(byte[] sendByte) //同期送信を始めます。(2 検証済)
    {
        string targetIP = sendIP;
        int port = sendPort;

        if (udpClientSend == null) udpClientSend = new UdpClient(new IPEndPoint(IPAddress.Parse(ScanIPAddr.IP[0]),GetSendHostPort()));

        udpClientSend.EnableBroadcast = true;
        Socket uSocket = udpClientSend.Client;
        uSocket.SetSocketOption(SocketOptionLevel.Socket,SocketOptionName.Broadcast, 1);

        if (targetIP == null)
        {
            udpClientSend.Send(sendByte, sendByte.Length, new IPEndPoint(IPAddress.Broadcast, sendPort));
            Debug.Log("送信処理しました。" + ScanIPAddr.IP[0] + " > BroadCast " + IPAddress.Broadcast + ":" + sendPort);
        }
        else
        {
            udpClientSend.Send(sendByte, sendByte.Length, new IPEndPoint(IPAddress.Parse(targetIP), sendPort));
            Debug.Log("送信処理しました。" + ScanIPAddr.IP[0] + " > " + IPAddress.Parse(targetIP) + ":" + sendPort);
        }
    }
    public void Send(byte[] sendByte,byte retryCount = 0) //非同期送信をUdpClientで開始します。(通常) <retry>
    {
        string targetIP = sendIP;
        int port = sendPort;

        if (sendTaskCount > 0)//送信中タスクの確認。 送信中有の場合、定数時間後リトライ
        {

            Debug.Log("SendTask is There.["+retryCount);
            retryCount++;

            if (retryCount > 10)
            {
                Debug.LogError("Retry OverFlow.");
                return;
            }

            Timer timer = new Timer(RETRY_SEND_TIME);
            timer.Elapsed += delegate (object obj, ElapsedEventArgs e) { Send(sendByte,retryCount); timer.Stop(); };
            timer.Start();
            return;
        }
        sendTaskCount++; //送信中タスクを増加

        if (udpClientSend == null) ;
        udpClientSend = new UdpClient(new IPEndPoint(IPAddress.Parse(ScanIPAddr.IP[0]), GetSendHostPort()));

        if (targetIP == null)
        {
            udpClientSend.BeginSend(sendByte, sendByte.Length, new IPEndPoint(IPAddress.Broadcast, sendPort),UDPSender,udpClientSend);
            Debug.Log("送信処理しました。" + ScanIPAddr.IP[0] + " > BroadCast " + IPAddress.Broadcast + ":" + sendPort);
        }
        else
        {
            udpClientSend.BeginSend(sendByte, sendByte.Length, sendIP, sendPort, UDPSender, udpClientSend);
            Debug.Log("送信処理しました。" + ScanIPAddr.IP[0] + " > " + IPAddress.Parse(targetIP) + ":" + sendPort + "["+sendByte[0]+"]["+sendByte[1]+"]...");
        }
    }

    void UDPSender(IAsyncResult res)
    {
        UdpClient udp = (UdpClient) res.AsyncState;
        try
        {
            udp.EndSend(res);
            Debug.Log("Send");
        }
        catch (SocketException ex)
        {
            Debug.Log("Error" + ex);
            return;
        }
        catch (ObjectDisposedException) // Finish : Socket Closed
        {
            Debug.Log("Socket Already Closed.");
            return;
        }

        sendTaskCount--;
        udp.Close();

    }
   

}

public class ScanIPAddr
{
    public static string[] IP { get { return Get(); } }
    public static byte[][] ByteIP { get { return GetByte(); } }

    public static string[] Get()
    {
        IPAddress[] addr_arr = Dns.GetHostAddresses(Dns.GetHostName());
        List<string> list = new List<string>();
        foreach (IPAddress address in addr_arr)
        {
            if (address.AddressFamily == System.Net.Sockets.AddressFamily.InterNetwork)
            {
                list.Add(address.ToString());
            }
        }
        if (list.Count == 0) return null;
        return list.ToArray();
    }
    public static byte[][] GetByte()
    {
        IPAddress[] addr_arr = Dns.GetHostAddresses(Dns.GetHostName());
        List<byte[]> list = new List<byte[]>();
        foreach (IPAddress address in addr_arr)
        {
            if (address.AddressFamily == System.Net.Sockets.AddressFamily.InterNetwork)
            {
                list.Add(address.GetAddressBytes());
            }
        }
        if (list.Count == 0) return null;
        return list.ToArray();
    }
}
public class ScanDevice
{
    public static int[] UsePort { get { return GetUsedPort(); } }

    static public bool CanUsePortChack(int port)//使用済みポートか確認します。使用可能 > T
    {
        bool b = true;
        foreach(int i in GetUsedPort())
        {
            if (i == port) b = false;
        }
        return b;
    }
    private static int[] GetUsedPort()
    {
        var v = System.Net.NetworkInformation.IPGlobalProperties.GetIPGlobalProperties();
        var v2 = v.GetActiveUdpListeners();
        int[] r = new int[v2.Length];

        int i = 0;
        foreach (var n in v2)
        {
            r[i] = n.Port;
            i++;
        }
        return r;
    }
}