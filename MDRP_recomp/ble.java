package com.sony.mocopi.datareader.ble;

import android.annotation.SuppressLint;
import android.bluetooth.BluetoothAdapter;
import android.bluetooth.BluetoothDevice;
import android.bluetooth.BluetoothGatt;
import android.bluetooth.BluetoothGattCallback;
import android.bluetooth.BluetoothGattCharacteristic;
import android.bluetooth.BluetoothGattDescriptor;
import android.bluetooth.BluetoothGattService;
import android.bluetooth.BluetoothManager;
import android.bluetooth.le.BluetoothLeScanner;
import android.bluetooth.le.ScanCallback;
import android.bluetooth.le.ScanFilter;
import android.bluetooth.le.ScanResult;
import android.bluetooth.le.ScanSettings;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.os.Build;
import android.os.Handler;
import android.os.Looper;
import androidx.annotation.RequiresApi;
import com.unity3d.player.UnityPlayer;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import java.util.Timer;
import java.util.TimerTask;
import java.util.UUID;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.regex.Pattern;

/* loaded from: classes.jar:com/sony/mocopi/datareader/ble/BLEManager.class */
public class BLEManager {
    private BluetoothManager mBluetoothManager;
    private BluetoothLeScanner mBluetoothLeScanner;
    private ScanCallback mScanCallback;
    private Context mContext;
    private String mTargetGameObject;
    private Intent mBluetoothStateIntent;
    private final String TAG = "BLEManager";
    private Set<BluetoothGatt> mBluetoothGattList = new LinkedHashSet();
    private boolean isScanning = false;
    private boolean mIsBusy = false;
    private final long CONNECTION_TIMEOUT = 2000;
    private Timer mQueueTimer = null;
    private final Handler mQueueHandler = new Handler(Looper.getMainLooper());
    private final ConcurrentLinkedQueue<String> eventQueue = new ConcurrentLinkedQueue<>();

    @SuppressLint({"MissingPermission"})
    private final BluetoothGattCallback mGattCallback = new BluetoothGattCallback() { // from class: com.sony.mocopi.datareader.ble.BLEManager.2
        @Override // android.bluetooth.BluetoothGattCallback
        public void onConnectionStateChange(BluetoothGatt gatt, int status, int newState) {
            if (status != 0) {
                BLEManager.this.mIsBusy = false;
                BLEManager.this.mBluetoothGattList.remove(gatt);
                gatt.disconnect();
                gatt.close();
                BLEManager.this.unitySender("onConnectionStateChange", String.valueOf(0), gatt.getDevice().getAddress(), gatt.getDevice().getName());
                return;
            }
            if (newState == 2) {
                gatt.discoverServices();
            } else if (newState == 0) {
                BLEManager.this.mBluetoothGattList.remove(gatt);
                gatt.close();
                BLEManager.this.unitySender("onConnectionStateChange", String.valueOf(0), gatt.getDevice().getAddress(), gatt.getDevice().getName());
            }
        }

        @Override // android.bluetooth.BluetoothGattCallback
        public void onServicesDiscovered(BluetoothGatt gatt, int status) {
            if (status == 0) {
                BLEManager.this.unitySender("onConnectionStateChange", String.valueOf(2), gatt.getDevice().getAddress(), gatt.getDevice().getName());
                BLEManager.this.unitySender("onServicesDiscovered", gatt.getDevice().getAddress(), gatt.getDevice().getName());
            }
        }

        @Override // android.bluetooth.BluetoothGattCallback
        public void onCharacteristicRead(BluetoothGatt gatt, BluetoothGattCharacteristic characteristic, int status) {
            BLEManager.this.mIsBusy = false;
            if (status == 0) {
                byte[] readBytes = characteristic.getValue();
                if (characteristic.getUuid().equals(UUID.fromString(Definition.UUID_BATTERY_CHAR))) {
                    int value = readBytes[0] & 255;
                    BLEManager.this.unitySender("onCharacteristicChanged", "Battery", gatt.getDevice().getAddress(), gatt.getDevice().getName(), String.valueOf(value));
                }
            }
        }

        @Override // android.bluetooth.BluetoothGattCallback
        public void onCharacteristicWrite(BluetoothGatt gatt, BluetoothGattCharacteristic characteristic, int status) {
            BLEManager.this.mIsBusy = false;
        }

        @Override // android.bluetooth.BluetoothGattCallback
        public void onCharacteristicChanged(BluetoothGatt gatt, BluetoothGattCharacteristic characteristic) {
            byte[] byteChara = characteristic.getValue();
            if (characteristic.getUuid().equals(UUID.fromString(Definition.UUID_SENSOR_CHAR))) {
                BLEManager.this.parseAndSendSensorData(gatt, byteChara);
            } else {
                if (characteristic.getUuid().equals(UUID.fromString(Definition.UUID_CMD_RESPONSE_CHAR))) {
                }
            }
        }

        @Override // android.bluetooth.BluetoothGattCallback
        public void onDescriptorWrite(BluetoothGatt gatt, BluetoothGattDescriptor descriptor, int status) {
            BLEManager.this.mIsBusy = false;
        }
    };
    private final BroadcastReceiver mBluetoothStateReceiver = new BroadcastReceiver() { // from class: com.sony.mocopi.datareader.ble.BLEManager.3
        @Override // android.content.BroadcastReceiver
        public void onReceive(Context context, Intent intent) {
            String action = intent.getAction();
            if ("android.bluetooth.adapter.action.STATE_CHANGED".equals(action)) {
                int state = intent.getIntExtra("android.bluetooth.adapter.extra.STATE", Integer.MIN_VALUE);
                switch (state) {
                    case 10:
                        BLEManager.this.unitySender("onUpdateState", "PoweredOff");
                        return;
                    case 12:
                        BLEManager.this.unitySender("onUpdateState", "PoweredOn");
                        return;
                    default:
                        return;
                }
            }
        }
    };

    public void initialize(String targetGameObject) {
        this.mTargetGameObject = targetGameObject;
        this.mContext = UnityPlayer.currentActivity;
        if (!this.mContext.getPackageManager().hasSystemFeature("android.hardware.bluetooth_le")) {
            unitySender("onUpdateState", "Unsupported");
            return;
        }
        if (!checkBLEPermission()) {
            unitySender("onUpdateState", "PermissionDenied");
            return;
        }
        this.mBluetoothManager = (BluetoothManager) this.mContext.getSystemService("bluetooth");
        BluetoothAdapter adapter = this.mBluetoothManager.getAdapter();
        if (adapter == null) {
            unitySender("onUpdateState", "Unsupported");
            return;
        }
        if (adapter.isEnabled()) {
            unitySender("onUpdateState", "PoweredOn");
        } else {
            unitySender("onUpdateState", "PoweredOff");
        }
        registerReceiver();
        startQueueTimer();
    }

    @SuppressLint({"MissingPermission"})
    public void startScan() {
        if (checkBluetoothAdapter() && checkBLEPermission() && !this.isScanning) {
            ScanSettings.Builder scanSettings = new ScanSettings.Builder();
            scanSettings.setScanMode(1);
            ScanSettings settings = scanSettings.build();
            this.mScanCallback = initScanCallbacks();
            this.mBluetoothLeScanner = this.mBluetoothManager.getAdapter().getBluetoothLeScanner();
            this.mBluetoothLeScanner.startScan((List<ScanFilter>) null, settings, this.mScanCallback);
            this.isScanning = true;
        }
    }

    @SuppressLint({"MissingPermission"})
    public void stopScan() {
        if (checkBluetoothAdapter() && checkBLEPermission() && this.isScanning) {
            this.mBluetoothLeScanner.stopScan(this.mScanCallback);
            this.isScanning = false;
        }
    }

    public boolean isScanning() {
        return this.isScanning;
    }

    private ScanCallback initScanCallbacks() {
        return new ScanCallback() { // from class: com.sony.mocopi.datareader.ble.BLEManager.1
            @Override // android.bluetooth.le.ScanCallback
            @SuppressLint({"MissingPermission"})
            public void onScanResult(int callbackType, ScanResult result) {
                super.onScanResult(callbackType, result);
                BluetoothDevice device = result.getDevice();
                String deviceName = device.getName();
                if (deviceName != null && Pattern.matches("QM-SS1 [A-Z0-9]{5}", deviceName)) {
                    String address = result.getDevice().getAddress();
                    String name = result.getDevice().getName();
                    BLEManager.this.unitySender("onScanResult", address, name);
                }
            }

            @Override // android.bluetooth.le.ScanCallback
            public void onBatchScanResults(List<ScanResult> results) {
                super.onBatchScanResults(results);
            }

            @Override // android.bluetooth.le.ScanCallback
            public void onScanFailed(int errorCode) {
                super.onScanFailed(errorCode);
            }
        };
    }

    @SuppressLint({"MissingPermission"})
    public boolean connect(String address) {
        if (!BluetoothAdapter.checkBluetoothAddress(address) || !checkBluetoothAdapter() || !checkBLEPermission() || this.mBluetoothGattList.size() >= 6) {
            return false;
        }
        BluetoothAdapter bluetoothAdapter = this.mBluetoothManager.getAdapter();
        BluetoothDevice device = bluetoothAdapter.getRemoteDevice(address);
        if (device == null) {
            return false;
        }
        BluetoothGatt connectedGatt = findGattByAddress(address);
        if (connectedGatt != null) {
            return false;
        }
        int state = this.mBluetoothManager.getConnectionState(device, 7);
        if (state != 0) {
            return false;
        }
        BluetoothGatt bluetoothGatt = device.connectGatt(this.mContext, false, this.mGattCallback, 2);
        bluetoothGatt.requestConnectionPriority(1);
        this.mBluetoothGattList.add(bluetoothGatt);
        return true;
    }

    @SuppressLint({"MissingPermission"})
    public void disconnect(String address) {
        BluetoothGatt bluetoothGatt;
        if (!BluetoothAdapter.checkBluetoothAddress(address) || !checkBluetoothAdapter() || !checkBLEPermission() || (bluetoothGatt = findGattByAddress(address)) == null) {
            return;
        }
        int state = this.mBluetoothManager.getConnectionState(bluetoothGatt.getDevice(), 7);
        if (state != 2) {
            return;
        }
        disconnectProcess(address);
    }

    @SuppressLint({"MissingPermission"})
    public void close() {
        for (BluetoothGatt gatt : this.mBluetoothGattList) {
            gatt.close();
        }
        this.mBluetoothGattList.clear();
        stopQueueTimer();
    }

    public void setup(String address) {
        String notifySensor = joinString("setNotification", address, Definition.UUID_SENSOR_SERVICE, Definition.UUID_SENSOR_CHAR, "true");
        if (!this.eventQueue.contains(notifySensor)) {
            this.eventQueue.add(notifySensor);
        }
        String notifyResponse = joinString("setNotification", address, Definition.UUID_CMD_SERVICE, Definition.UUID_CMD_RESPONSE_CHAR, "true");
        if (!this.eventQueue.contains(notifyResponse)) {
            this.eventQueue.add(notifyResponse);
        }
        String writeSetRTC = joinString("writeCharacteristic", address, Definition.UUID_CMD_SERVICE, Definition.UUID_CMD_CHAR, "setRtcCommand");
        if (!this.eventQueue.contains(writeSetRTC)) {
            this.eventQueue.add(writeSetRTC);
        }
        String writeSensor = joinString("writeCharacteristic", address, Definition.UUID_CMD_SERVICE, Definition.UUID_CMD_CHAR, "startSensor");
        if (!this.eventQueue.contains(writeSensor)) {
            this.eventQueue.add(writeSensor);
        }
    }

    @SuppressLint({"MissingPermission"})
    public boolean getBatteryLevel(String address) {
        BluetoothGatt bluetoothGatt;
        if (!BluetoothAdapter.checkBluetoothAddress(address) || !checkBluetoothAdapter() || !checkBLEPermission() || (bluetoothGatt = findGattByAddress(address)) == null) {
            return false;
        }
        int state = this.mBluetoothManager.getConnectionState(bluetoothGatt.getDevice(), 7);
        if (state != 2) {
            return false;
        }
        String batteryLevel = joinString("readCharacteristic", address, Definition.UUID_BATTERY_SERVICE, Definition.UUID_BATTERY_CHAR);
        if (!this.eventQueue.contains(batteryLevel)) {
            this.eventQueue.add(batteryLevel);
            return true;
        }
        return false;
    }

    public void registerReceiver() {
        if (this.mContext != null && this.mBluetoothStateIntent == null) {
            IntentFilter intentFilter = new IntentFilter("android.bluetooth.adapter.action.STATE_CHANGED");
            this.mBluetoothStateIntent = this.mContext.registerReceiver(this.mBluetoothStateReceiver, intentFilter);
        }
    }

    public void unregisterReceiver() {
        if (this.mContext == null || this.mBluetoothStateIntent == null) {
            return;
        }
        this.mContext.unregisterReceiver(this.mBluetoothStateReceiver);
    }

    /* JADX INFO: Access modifiers changed from: private */
    @SuppressLint({"MissingPermission"})
    public boolean readCharacteristic(String address, String service, String characteristic) {
        BluetoothGattService gattService;
        BluetoothGattCharacteristic gattCharacteristic;
        if (!BluetoothAdapter.checkBluetoothAddress(address) || !checkBluetoothAdapter() || !checkBLEPermission()) {
            return false;
        }
        UUID uuid_service = UUID.fromString(service);
        UUID uuid_characteristic = UUID.fromString(characteristic);
        BluetoothGatt bluetoothGatt = findGattByAddress(address);
        if (bluetoothGatt == null || (gattService = bluetoothGatt.getService(uuid_service)) == null || (gattCharacteristic = gattService.getCharacteristic(uuid_characteristic)) == null) {
            return false;
        }
        boolean result = bluetoothGatt.readCharacteristic(gattCharacteristic);
        if (!result) {
            return false;
        }
        this.mIsBusy = true;
        return true;
    }

    /* JADX INFO: Access modifiers changed from: private */
    @SuppressLint({"MissingPermission"})
    public boolean writeCharacteristic(String address, String service, String characteristic, byte[] val) {
        BluetoothGattService gattService;
        BluetoothGattCharacteristic gattCharacteristic;
        if (!BluetoothAdapter.checkBluetoothAddress(address) || !checkBluetoothAdapter()) {
            return false;
        }
        UUID uuid_service = UUID.fromString(service);
        UUID uuid_characteristic = UUID.fromString(characteristic);
        BluetoothGatt bluetoothGatt = findGattByAddress(address);
        if (bluetoothGatt == null || (gattService = bluetoothGatt.getService(uuid_service)) == null || (gattCharacteristic = gattService.getCharacteristic(uuid_characteristic)) == null) {
            return false;
        }
        if (Build.VERSION.SDK_INT >= 33) {
            int result = bluetoothGatt.writeCharacteristic(gattCharacteristic, val, 2);
            if (result != 0) {
                return false;
            }
        } else {
            gattCharacteristic.setValue(val);
            boolean result2 = bluetoothGatt.writeCharacteristic(gattCharacteristic);
            if (!result2) {
                return false;
            }
        }
        this.mIsBusy = true;
        return true;
    }

    /* JADX INFO: Access modifiers changed from: private */
    @SuppressLint({"MissingPermission"})
    public boolean setNotification(String address, String service, String characteristic, boolean enable) {
        BluetoothGattService gattService;
        BluetoothGattCharacteristic gattCharacteristic;
        if (!BluetoothAdapter.checkBluetoothAddress(address) || !checkBluetoothAdapter()) {
            return false;
        }
        UUID uuid_service = UUID.fromString(service);
        UUID uuid_characteristic = UUID.fromString(characteristic);
        BluetoothGatt bluetoothGatt = findGattByAddress(address);
        if (bluetoothGatt == null || (gattService = bluetoothGatt.getService(uuid_service)) == null || (gattCharacteristic = gattService.getCharacteristic(uuid_characteristic)) == null) {
            return false;
        }
        bluetoothGatt.setCharacteristicNotification(gattCharacteristic, enable);
        BluetoothGattDescriptor descriptor = gattCharacteristic.getDescriptor(UUID.fromString("00002902-0000-1000-8000-00805f9b34fb"));
        if (Build.VERSION.SDK_INT >= 33) {
            int result = bluetoothGatt.writeDescriptor(descriptor, BluetoothGattDescriptor.ENABLE_NOTIFICATION_VALUE);
            if (result != 0) {
                return false;
            }
        } else {
            descriptor.setValue(BluetoothGattDescriptor.ENABLE_NOTIFICATION_VALUE);
            boolean result2 = bluetoothGatt.writeDescriptor(descriptor);
            if (!result2) {
                return false;
            }
        }
        this.mIsBusy = true;
        return true;
    }

    /* JADX INFO: Access modifiers changed from: private */
    public BluetoothGatt findGattByAddress(String address) {
        if (this.mBluetoothGattList != null && address != null) {
            Set<BluetoothGatt> copyOfGattSet = new LinkedHashSet<>(this.mBluetoothGattList);
            for (BluetoothGatt gatt : copyOfGattSet) {
                if (gatt.getDevice().getAddress().equals(address)) {
                    return gatt;
                }
            }
            return null;
        }
        return null;
    }

    private boolean checkBluetoothAdapter() {
        BluetoothAdapter adapter;
        return (this.mBluetoothManager == null || (adapter = this.mBluetoothManager.getAdapter()) == null || !adapter.isEnabled()) ? false : true;
    }

    /* JADX INFO: Access modifiers changed from: private */
    @SuppressLint({"MissingPermission"})
    public void parseAndSendSensorData(BluetoothGatt gatt, byte[] data) {
        System.currentTimeMillis();
        int ROT_LEN = 2 * 4;
        int VEL_LEN = 2 * 3;
        for (int i = 0; i < 2; i++) {
            long time = (Utility.convertByteArrayToLong(data, 0) / 1000000) + (10 * i);
            float rot_x = Utility.getInt16(data, (i * ROT_LEN) + 8, ByteOrder.LITTLE_ENDIAN) / 8192;
            float rot_y = Utility.getInt16(data, ((i * ROT_LEN) + 8) + 2, ByteOrder.LITTLE_ENDIAN) / 8192;
            float rot_z = Utility.getInt16(data, ((i * ROT_LEN) + 8) + (2 * 2), ByteOrder.LITTLE_ENDIAN) / 8192;
            float rot_w = Utility.getInt16(data, ((i * ROT_LEN) + 8) + (2 * 3), ByteOrder.LITTLE_ENDIAN) / 8192;
            float acc_x = Utility.getFloat16(data, ((i * VEL_LEN) + 8) + (ROT_LEN * 2)) / 8.0f;
            float acc_y = Utility.getFloat16(data, (((i * VEL_LEN) + 8) + (ROT_LEN * 2)) + 2) / 8.0f;
            float acc_z = Utility.getFloat16(data, (((i * VEL_LEN) + 8) + (ROT_LEN * 2)) + (2 * 2)) / 8.0f;
            unitySender("onCharacteristicChanged", "Sensor", gatt.getDevice().getAddress(), gatt.getDevice().getName(), String.valueOf(time), String.valueOf(rot_x), String.valueOf(rot_y), String.valueOf(rot_z), String.valueOf(rot_w), String.valueOf(acc_x), String.valueOf(acc_y), String.valueOf(acc_z));
        }
    }

    private void disconnectProcess(String address) {
        String notifySensor = joinString("setNotification", address, Definition.UUID_SENSOR_SERVICE, Definition.UUID_SENSOR_CHAR, "false");
        if (!this.eventQueue.contains(notifySensor)) {
            this.eventQueue.add(notifySensor);
        }
        String notifyResponse = joinString("setNotification", address, Definition.UUID_CMD_SERVICE, Definition.UUID_CMD_RESPONSE_CHAR, "false");
        if (!this.eventQueue.contains(notifyResponse)) {
            this.eventQueue.add(notifyResponse);
        }
        String writeSensor = joinString("writeCharacteristic", address, Definition.UUID_CMD_SERVICE, Definition.UUID_CMD_CHAR, "stopSensor");
        if (!this.eventQueue.contains(writeSensor)) {
            this.eventQueue.add(writeSensor);
        }
        String disconnect = joinString("disconnect", address);
        if (!this.eventQueue.contains(disconnect)) {
            this.eventQueue.add(disconnect);
        }
    }

    private String joinString(String... params) {
        return String.join(",", params);
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void unitySender(String... params) {
        String param = String.join(",", params);
        UnityPlayer.UnitySendMessage(this.mTargetGameObject, "NativePluginReceiver", param);
    }

    private boolean checkBLEPermission() {
        if (Build.VERSION.SDK_INT >= 31) {
            return checkBLEPermissionS();
        }
        return checkBLEPermissionOld();
    }

    @RequiresApi(api = 31)
    private boolean checkBLEPermissionS() {
        int bt_connect = this.mContext.checkSelfPermission("android.permission.BLUETOOTH_CONNECT");
        int bt_scan = this.mContext.checkSelfPermission("android.permission.BLUETOOTH_SCAN");
        int finelocation = this.mContext.checkSelfPermission("android.permission.ACCESS_FINE_LOCATION");
        if (bt_connect != 0 || bt_scan != 0 || finelocation != 0) {
            return false;
        }
        return true;
    }

    private boolean checkBLEPermissionOld() {
        int finelocation = this.mContext.checkSelfPermission("android.permission.ACCESS_FINE_LOCATION");
        if (finelocation != 0) {
            return false;
        }
        return true;
    }

    /* JADX INFO: Access modifiers changed from: private */
    public byte[] setRTCCommandTime() {
        byte[] setRtcCommand = Definition.CMD_SET_RTC_MS;
        long currentTime = System.currentTimeMillis();
        ByteBuffer buffer = ByteBuffer.allocate(8);
        buffer.order(ByteOrder.LITTLE_ENDIAN);
        buffer.putLong(currentTime);
        byte[] timeToByteArray = buffer.array();
        System.arraycopy(timeToByteArray, 0, setRtcCommand, 4, timeToByteArray.length);
        return setRtcCommand;
    }

    @SuppressLint({"MissingPermission"})
    private void startQueueTimer() {
        if (this.mQueueTimer == null) {
            this.mQueueTimer = new Timer();
            this.mQueueTimer.schedule(new TimerTask() { // from class: com.sony.mocopi.datareader.ble.BLEManager.4
                @Override // java.util.TimerTask, java.lang.Runnable
                public void run() {
                    BLEManager.this.mQueueHandler.post(new Runnable() { // from class: com.sony.mocopi.datareader.ble.BLEManager.4.1
                        @Override // java.lang.Runnable
                        public void run() {
                            BluetoothGatt bluetoothGatt;
                            if (BLEManager.this.mIsBusy) {
                                return;
                            }
                            String event = BLEManager.this.eventQueue.isEmpty() ? null : (String) BLEManager.this.eventQueue.poll();
                            if (event != null) {
                                String[] param = event.split(",");
                                if (param[0].equals("setNotification")) {
                                    BLEManager.this.setNotification(param[1], param[2], param[3], Boolean.parseBoolean(param[4]));
                                    return;
                                }
                                if (param[0].equals("writeCharacteristic") && param[4].equals("setRtcCommand")) {
                                    byte[] value = BLEManager.this.setRTCCommandTime();
                                    BLEManager.this.writeCharacteristic(param[1], param[2], param[3], value);
                                    return;
                                }
                                if (param[0].equals("writeCharacteristic") && param[4].equals("startSensor")) {
                                    BLEManager.this.writeCharacteristic(param[1], param[2], param[3], Definition.CMD_START_SENSOR);
                                    return;
                                }
                                if (param[0].equals("writeCharacteristic") && param[4].equals("stopSensor")) {
                                    BLEManager.this.writeCharacteristic(param[1], param[2], param[3], Definition.CMD_STOP_SENSOR);
                                    return;
                                }
                                if (param[0].equals("readCharacteristic")) {
                                    BLEManager.this.readCharacteristic(param[1], param[2], param[3]);
                                } else {
                                    if (!param[0].equals("disconnect") || (bluetoothGatt = BLEManager.this.findGattByAddress(param[1])) == null) {
                                        return;
                                    }
                                    bluetoothGatt.disconnect();
                                }
                            }
                        }
                    });
                }
            }, 0L, 100L);
        }
    }

    private void stopQueueTimer() {
        if (this.mQueueTimer == null) {
            return;
        }
        this.mQueueTimer.cancel();
        this.mQueueTimer.purge();
        this.mQueueTimer = null;
    }
}