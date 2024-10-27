import asyncio
from bleak import BleakScanner, BleakClient

class MocopiSensorDataReader:
    def __init__(self):
        self.on_scan_result = None
        self.on_sensor_update = None
        self.on_connection_state_changed = None
        self.on_battery_level = None

    async def initialize(self):
        """ Initialize the Bluetooth scanner """
        print("Initializing BLE scanner...")
        self.scanner = BleakScanner()

    async def start_scan(self):
        """ Start scanning for Bluetooth devices """
        print("Starting scan...")
        devices = await self.scanner.discover()
        for device in devices:
            print(f"Found device: {device}")
            if self.on_scan_result:
                self.on_scan_result(device)

    async def connect(self, device_address):
        """ Connect to a device by its address """
        print(f"Connecting to {device_address}...")
        self.client = BleakClient(device_address)
        await self.client.connect()
        if self.client.is_connected:
            print(f"Connected to {device_address}")
            if self.on_connection_state_changed:
                self.on_connection_state_changed(device_address, "CONNECTED")

    async def disconnect(self):
        """ Disconnect the device """
        if self.client.is_connected:
            await self.client.disconnect()
            print("Disconnected")
            if self.on_connection_state_changed:
                self.on_connection_state_changed(self.client.address, "DISCONNECTED")

    async def get_battery_level(self):
        """ Get battery level from the connected device """
        if self.client.is_connected:
            try:
                # 標準的なバッテリーレベルUUIDを使用
                battery_level = await self.client.read_gatt_char("00002a19-0000-1000-8000-00805f9b34fb")
                if self.on_battery_level:
                    # bytearray から整数に変換
                    battery_level_int = int(battery_level[0])
                    self.on_battery_level(self.client.address, battery_level_int)
                print(f"Battery Level: {battery_level_int}")
            except Exception as e:
                print(f"Failed to get battery level: {e}")
        else:
            print("Device not connected")


    async def on_sensor_data_received(self, sender, data):
        """ Handle sensor data (e.g., accelerometer/gyroscope) """
        print(f"Sensor data received from {sender}: {data}")
        if self.on_sensor_update:
            self.on_sensor_update(sender, data)

    def register_callbacks(self, scan_callback, sensor_callback, connection_callback, battery_callback):
        """ Register the event handlers """
        self.on_scan_result = scan_callback
        self.on_sensor_update = sensor_callback
        self.on_connection_state_changed = connection_callback
        self.on_battery_level = battery_callback


async def main():
    reader = MocopiSensorDataReader()

    def handle_scan_result(device):
        print(f"Device found: {device}")

    def handle_sensor_update(device, data):
        print(f"Sensor data from {device}: {data}")

    def handle_connection_state_change(device, state):
        print(f"Connection state changed for {device}: {state}")

    def handle_battery_level(device, level):
        print(f"Battery level for {device}: {level}")

    reader.register_callbacks(handle_scan_result, handle_sensor_update, handle_connection_state_change, handle_battery_level)

    await reader.initialize()
    await reader.start_scan()
    # Replace with actual device address you want to connect
    await reader.connect("3C:38:F4:B4:35:BB")  
    await reader.get_battery_level()
    await asyncio.sleep(5)
    await reader.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
