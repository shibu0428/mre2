import asyncio
from bleak import BleakClient, BleakScanner, BleakError

async def list_services(device_address):
    client = BleakClient(device_address)
    try:
        await asyncio.wait_for(client.connect(), timeout=10)  # 10秒のタイムアウトを設定
        services = await client.get_services()
        for service in services:
            print(f"Service: {service.uuid}")
            for characteristic in service.characteristics:
                print(f"  Characteristic: {characteristic.uuid}, Properties: {characteristic.properties}")
        await client.disconnect()
    except asyncio.TimeoutError:
        print(f"Connection to {device_address} timed out.")
    except BleakError as e:
        print(f"Failed to connect: {e}")
    except AttributeError as e:
        print(f"An attribute error occurred: {e}")

async def main():
    # スキャンしてデバイスを見つける
    devices = await BleakScanner.discover()
    for i, device in enumerate(devices):
        print(f"{i}: {device.name} ({device.address})")

    # 接続したいデバイスの番号を選ぶ
    device_number = int(input("\nSelect device by number: "))
    device_address = devices[device_number].address

    # サービスとキャラクタリスティックを表示
    await list_services(device_address)

if __name__ == "__main__":
    asyncio.run(main())
