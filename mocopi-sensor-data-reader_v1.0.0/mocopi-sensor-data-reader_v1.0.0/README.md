# mocopiSensorDataReader

Unity 製 Android / iOS アプリケーションから mocopi 単体に接続してセンサーデータを取得することができるプラグインです。


# 機能

* 周辺の mocopi 検出
* mocopi との接続
* mocopi のセンサーデータ ( Velocity, Rotation ) の取得
* mocopi の電池残量の取得


# EULA
SDK同梱のEULAに従ってご使用いただけます。SDKご使用開始前にご一読ください。
また、SDKの使用開始をもって、本契約にご同意いただいたものとします 。


# 環境

* Unity 2022.3.13f1 以降
* Android 11 以降
* iOS 17.0 以降

# インストール

* Assets -> Import Package から MocopiSensorDataReader.unitypackage をインポートしてください。


# 使用方法

mocopiSensorDataReader -> Resources -> Prefabs -> mocopiSensorDataReaderPrefab をシーン上に配置してください。



**初期化**

始めに初期化する必要があります。  
初期化の結果を OnUpdateState で受け取ります。  
※備考※  
OnUpdateState は スマートフォンの Bluetooth 機能の ON / OFF 切替でも呼ばれます。

```
var mocopiSensorDataReader = MocopiSensorDataReader.Instance;
mocopiSensorDataReader.OnUpdateState += OnUpdateState;
mocopiSensorDataReader.Initialize();
```
**結果**

初期化した結果、 UpdateState が「POWERED_ON」の時のみ以降の処理が使用できるようになります。

```
void OnUpdateState(MocopiSensorDataReader.UpdateState updateState)
{
    if(updateState == MocopiSensorDataReader.UpdateState.POWERED_ON){
        /* 利用可能 */
    }
}
```


**スキャン**

周辺の mocopi をスキャンします。スキャンの結果、発見した mocopi を OnScanResult で受け取ります。  
また、スキャンは自動的に停止しないので、利用者側で停止させる必要があります。

```
var mocopiSensorDataReader = MocopiSensorDataReader.Instance;
mocopiSensorDataReader.OnScanResult += OnScanResult;
mocopiSensorDataReader.StartScan();
```
```
MocopiSensorDataReader.Instance.StopScan();
```

**結果**

スキャンの結果、同一 mocopi が多重に通知される事がありますので、利用者側で適宜重複を排除する等の処理を実施してください。

```
void OnScanResult(Device device)
{
    /* スキャン結果の保持 */
}
```

**接続**

スキャンの結果発見された mocopi に対して接続を試行します。  
接続要求に成功した場合、接続状態の変化を OnConnectionStateChanged で受け取れます。  
また、正常に接続処理が完了した場合は、センサー情報を OnSensorUpdate で受け取れます。  
接続要求に失敗した場合、コールバックはありません。  
最大同時接続台数は 6 台になっており、7 台以上接続しようとすると常に失敗を返します。

```
var mocopiSensorDataReader = MocopiSensorDataReader.Instance;
mocopiSensorDataReader.OnSensorUpdate += OnSensorUpdate;
mocopiSensorDataReader.OnConnectionStateChanged += OnConnectionStateChanged;
var result = mocopiSensorDataReader.Connect(/* OnScanResult で受け取ったデバイス*/);
if(!result){
    /* 接続要求に失敗 */
}
```
**結果**

```
void OnConnectionStateChanged(Device device, MocopiSensorDataReader.ConnectionState state)
{
    /* 接続・切断時の処理 */
}

void OnSensorUpdate(SensorData sensorData)
{
    /* 各デバイスのセンサー情報を使用した任意の処理 */
}
```
**電池残量取得**  
接続済みの mocopi に対して電池残量の取得要求を試行します。  
要求に成功した場合、電池残量を OnBatteryLevel で受け取れます。  
一回の取得要求に対して一回のコールバックがあります。  
取得要求に失敗した場合、コールバックはありません。 

```
mocopiSensorDataReader = MocopiSensorDataReader.Instance;
mocopiSensorDataReader.OnBatteryLevel += OnBatteryLevel;
var result = mocopiSensorDataReader.GetBatteryLevel(/* OnScanResult で受け取ったデバイス*/);
if(!result){
    /* 電池残量取得要求に失敗 */
}
```
**結果**

```
void OnBatteryLevel(Device device, int batteryLevel)
{
　　/* 各デバイスの電池残量を使用した任意の処理 */
}
```
# Android向け
BluetoothLE 機能を使用しているので、  
AndroidManifest.xml にパーミッションを追加する必要があります。

```
<manifest>
    <uses-permission android:name="android.permission.BLUETOOTH"
                     android:maxSdkVersion="30" />
    <uses-permission android:name="android.permission.BLUETOOTH_ADMIN"
                     android:maxSdkVersion="30" />

    <uses-permission android:name="android.permission.BLUETOOTH_SCAN" />
    <uses-permission android:name="android.permission.BLUETOOTH_CONNECT" />

    <uses-permission android:name="android.permission.ACCESS_FINE_LOCATION" />
    ...
</manifest>
```

# iOS向け
BluetoothLE 機能を使用しているので、  
Info.plist に Privacy - Bluetooth Always Usage Description を追加し、value に使用目的を記述する必要があります。

# 動作確認機種
* Xperia 1 III ( Android 13 )
* iPhone 15 Pro ( iOS 17.4 )

