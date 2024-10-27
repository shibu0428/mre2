using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI; // UIコンポーネントの使用
using System; 
using TMPro;

public class main : MonoBehaviour
{
    public GameObject gameObject; // 送信するオブジェクト（例: カメラやプレイヤー）
    public Text text;             // 画面に座標を表示するテキスト
    public TMPro.TMP_Text ipInputField; // IPアドレス入力用のUIテキストボックス
    public Button startButton;    // 送信開始ボタン

    Vector3 vector3 = new Vector3(0, 0, 0); // 送信する座標データ
    Quaternion quaternion = new Quaternion(0, 0, 0, 1); // 送信する回転データ
    UDPSystem udpSystem;  // UDP通信システム
    string ipAddr;        // 送信先のIPアドレス
    int sendPort = 5002;  // 送信先のポート番号

    private void Awake()
    {
        // ボタンに押したときの処理を登録
        startButton.onClick.AddListener(OnStartButtonClicked);
    }

    // ボタンが押されたときに呼ばれるメソッド
    void OnStartButtonClicked()
    {
        // 入力されたIPアドレスを取得
        ipAddr = ipInputField.text;

        // 入力されたIPアドレスが正しいかを確認
        if (string.IsNullOrEmpty(ipAddr))
        {
            Debug.LogError("IPアドレスが入力されていません。");
            return;
        }

        // UDPSystemのインスタンスを作成し、送信先IPとポートを設定
        udpSystem = new UDPSystem();
        udpSystem.SetSend(ipAddr, sendPort);

        Debug.Log("送信を開始します。IPアドレス: " + ipAddr);
    }

    // Update is called once per frame
    void Update()
    {
        // 送信処理：ボタンを押すと送信が開始され、毎フレーム座標と回転データを送信
        if (udpSystem != null)
        {
            vector3 = gameObject.transform.position;        // オブジェクトの位置を取得
            quaternion = gameObject.transform.rotation;     // オブジェクトの回転（クォータニオン）を取得

            DATA sendData = new DATA(vector3, quaternion);  // 座標と回転データをUDP用の形式に変換
            udpSystem.Send(sendData.ToByte(), 99);          // データを送信
            
            // 現在の座標と回転をUI上に表示
            text.text = "Pos: (" + vector3.x + "," + vector3.y + "," + vector3.z + ")" +
                        "\nRot: (" + quaternion.w + "," + quaternion.x + "," + quaternion.y + "," + quaternion.z + ")";
        }
    }
}

public class DATA
{
    private float x;
    private float y;
    private float z;
    private float qw;
    private float qx;
    private float qy;
    private float qz;

    // バイト配列から座標とクォータニオンデータを復元
    public DATA(byte[] bytes)
    {
        x = BitConverter.ToSingle(bytes, 0);
        y = BitConverter.ToSingle(bytes, 4);
        z = BitConverter.ToSingle(bytes, 8);
        qw = BitConverter.ToSingle(bytes, 12);
        qx = BitConverter.ToSingle(bytes, 16);
        qy = BitConverter.ToSingle(bytes, 20);
        qz = BitConverter.ToSingle(bytes, 24);
    }

    // 座標とクォータニオンデータをバイト配列に変換
    public DATA(Vector3 vector3, Quaternion quaternion)
    {
        x = vector3.x;
        y = vector3.y;
        z = vector3.z;
        qw = quaternion.w;
        qx = quaternion.x;
        qy = quaternion.y;
        qz = quaternion.z;
    }

    // 座標とクォータニオンをバイト配列に変換
    public byte[] ToByte()
    {
        byte[] xBytes = BitConverter.GetBytes(this.x);
        byte[] yBytes = BitConverter.GetBytes(this.y);
        byte[] zBytes = BitConverter.GetBytes(this.z);
        byte[] qwBytes = BitConverter.GetBytes(this.qw);
        byte[] qxBytes = BitConverter.GetBytes(this.qx);
        byte[] qyBytes = BitConverter.GetBytes(this.qy);
        byte[] qzBytes = BitConverter.GetBytes(this.qz);

        // バイト配列を結合して1つの配列にする
        return MargeByte(MargeByte(MargeByte(MargeByte(MargeByte(xBytes, yBytes), zBytes), qwBytes), qxBytes), MargeByte(qyBytes, qzBytes));
    }

    // バイト配列を結合するヘルパーメソッド
    public static byte[] MargeByte(byte[] baseByte, byte[] addByte)
    {
        byte[] b = new byte[baseByte.Length + addByte.Length];
        for (int i = 0; i < b.Length; i++)
        {
            if (i < baseByte.Length) b[i] = baseByte[i];
            else b[i] = addByte[i - baseByte.Length];
        }
        return b;
    }
}
