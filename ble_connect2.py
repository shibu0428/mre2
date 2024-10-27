import struct
import asyncio
from bleak import BleakClient, BleakScanner

class Definition:
    UUID_CMD_CHAR = "0000ff01-0000-1000-8000-00805f9b34fb"
    UUID_SENSOR_CHAR = "25047e64-657c-4856-afcf-e315048a965b"
    CMD_START_SENSOR = bytearray([126, 3, 24, 214, 1, 0, 0])  # センサーを開始するコマンド
    CMD_STOP_SENSOR = bytearray([126, 3, 24, 214, 0, 0, 0])   # センサーを停止するコマンド

# センサーデータを解析するコールバック
async def sensor_data_callback(sender, data):
    print(f"Notification received from {sender}")
    print(f"Raw sensor data: {data}")

    if len(data) < 24:
        print("Insufficient data length")
        return
    
    # タイムスタンプ (最初の8バイト)
    timestamp = struct.unpack_from('<Q', data, 0)[0] // 1000000
    print(f"Timestamp: {timestamp}")

    for i in range(2):
        base_index = 8 + i * 8
        
        # 回転データ (Quaternion) - 16ビット整数を浮動小数点数に変換
        rot_x = struct.unpack_from('<h', data, base_index)[0] / 8192
        rot_y = struct.unpack_from('<h', data, base_index + 2)[0] / 8192
        rot_z = struct.unpack_from('<h', data, base_index + 4)[0] / 8192
        rot_w = struct.unpack_from('<h', data, base_index + 6)[0] / 8192

        # 加速度データ (Acceleration) - 16ビット整数を浮動小数点数に変換
        acc_x = struct.unpack_from('<h', data, base_index + 16)[0] / 8.0
        acc_y = struct.unpack_from('<h', data, base_index + 18)[0] / 8.0
        acc_z = struct.unpack_from('<h', data, base_index + 20)[0] / 8.0

        print(f"Rotation - x: {rot_x}, y: {rot_y}, z: {rot_z}, w: {rot_w}")
        print(f"Acceleration - x: {acc_x}, y: {acc_y}, z: {acc_z}")

# デバイスのサービスとキャラクタリスティックを列挙する関数
async def list_services(client):
    services = await client.get_services()
    for service in services:
        print(f"Service: {service.uuid}")
        for char in service.characteristics:
            print(f"  Characteristic: {char.uuid}, Properties: {char.properties}")

# センサーを開始するコマンドを送信
async def send_start_sensor_command(client):
    await client.write_gatt_char(Definition.UUID_CMD_CHAR, Definition.CMD_START_SENSOR)
    print("Sent start sensor command")

# センサーを停止するコマンドを送信
async def send_stop_sensor_command(client):
    await client.write_gatt_char(Definition.UUID_CMD_CHAR, Definition.CMD_STOP_SENSOR)
    print("Sent stop sensor command")

# メインロジック（センサーデータの通知を取得するためのもの）
async def main():
    devices = await BleakScanner.discover()
    mocopi_device = None
    for device in devices:
        if device.name and "QM-SS1" in device.name:
            mocopi_device = device
            print(f"Found Mocopi device: {device.name}, {device.address}")
            break

    if mocopi_device is None:
        print("Mocopi device not found")
        return

    async with BleakClient(mocopi_device.address) as client:
        # デバイスのすべてのサービスとキャラクタリスティックを列挙
        await list_services(client)

        await asyncio.sleep(1)  # 1秒待機してから通知を開始
        await client.start_notify(Definition.UUID_SENSOR_CHAR, sensor_data_callback)
        print("Sensor notifications started")

        # センサーを開始するコマンドを送信
        await send_start_sensor_command(client)

        await asyncio.sleep(10)  # 10秒間センサーデータを受信

        # センサーを停止するコマンドを送信
        await send_stop_sensor_command(client)

        await client.stop_notify(Definition.UUID_SENSOR_CHAR)
        print("Sensor notifications stopped")

if __name__ == "__main__":
    asyncio.run(main())
