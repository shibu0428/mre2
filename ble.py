import asyncio
from bleak import BleakScanner

async def scan_devices():
    print("スキャンを開始します...")
    devices = await BleakScanner.discover()  # BLEデバイスを検索
    for device in devices:
        print(f"デバイス: {device.name}, アドレス: {device.address}")

# イベントループでスキャンを実行
asyncio.run(scan_devices())
