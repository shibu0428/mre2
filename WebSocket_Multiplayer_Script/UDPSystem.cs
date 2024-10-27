using System.Net.Sockets;
using System.Net;
using UnityEngine;
using System.Timers;
using System;

public class UDPSystem
{
    readonly int RETRY_SEND_TIME = 10; // ms

    static byte sendTaskCount = 0;

    string sendIP;
    int sendPort = 5000;

    UdpClient udpClientSend;

    // コンストラクタ1: 送信側の初期化
    public UDPSystem() { }

    // コンストラクタ2: 送信先のIPアドレスとポート番号を指定して初期化
    public UDPSystem(string send_ip, int sendport)
    {
        sendIP = send_ip;
        sendPort = sendport;
    }

    // 送信先IPアドレスとポート番号を設定する
    public void SetSend(string send_ip, int sendport)
    {
        sendIP = send_ip;
        sendPort = sendport;
    }

    // 非同期でデータを送信するメソッド (通常の送信)
    public void Send(byte[] sendByte, byte retryCount = 0)
    {
        string targetIP = sendIP;
        int port = sendPort;

        // 送信タスクがあればリトライ処理を実行
        if (sendTaskCount > 0)
        {
            Debug.Log("送信タスクが実行中です。リトライカウント: " + retryCount);
            retryCount++;
            if (retryCount > 10)
            {
                Debug.LogError("リトライ回数が超過しました。");
                return;
            }

            Timer timer = new Timer(RETRY_SEND_TIME);
            timer.Elapsed += delegate (object obj, ElapsedEventArgs e) { Send(sendByte, retryCount); timer.Stop(); };
            timer.Start();
            return;
        }

        sendTaskCount++; // 送信中タスクをカウント

        // UdpClientが未初期化の場合、初期化する
        if (udpClientSend == null)
        {
            udpClientSend = new UdpClient(new IPEndPoint(IPAddress.Any, 0)); // 自動的に空いているポートを使用
        }

        // 送信処理: ブロードキャストまたは特定のIPに送信
        if (targetIP == null)
        {
            udpClientSend.BeginSend(sendByte, sendByte.Length, new IPEndPoint(IPAddress.Broadcast, sendPort), UDPSender, udpClientSend);
            Debug.Log("送信しました: ブロードキャスト " + IPAddress.Broadcast + ":" + sendPort);
        }
        else
        {
            udpClientSend.BeginSend(sendByte, sendByte.Length, targetIP, sendPort, UDPSender, udpClientSend);
            Debug.Log("送信しました: " + targetIP + ":" + sendPort + " データ長: " + sendByte.Length);
        }
    }

    // 非同期送信完了時に呼ばれるコールバックメソッド
    void UDPSender(IAsyncResult res)
    {
        UdpClient udp = (UdpClient)res.AsyncState;
        try
        {
            udp.EndSend(res); // 送信完了処理
            Debug.Log("送信が完了しました。");
        }
        catch (SocketException ex)
        {
            Debug.Log("送信エラー: " + ex);
        }
        catch (ObjectDisposedException)
        {
            Debug.Log("ソケットが既に閉じられています。");
        }

        sendTaskCount--; // 送信タスクを減らす
        udp.Close();     // ソケットを閉じる
    }
}
