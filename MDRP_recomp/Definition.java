package com.sony.mocopi.datareader.ble;

/* loaded from: classes.jar:com/sony/mocopi/datareader/ble/Definition.class */
public class Definition {
    public static final String UUID_CMD_SERVICE = "0000ff00-0000-1000-8000-00805f9b34fb";
    public static final String UUID_SENSOR_SERVICE = "91a7608d-4456-479d-b9b1-4706e8711cf8";
    public static final String UUID_BATTERY_SERVICE = "0000180F-0000-1000-8000-00805f9b34fb";
    public static final String UUID_CMD_CHAR = "0000ff01-0000-1000-8000-00805f9b34fb";
    public static final String UUID_CMD_RESPONSE_CHAR = "0000ff03-0000-1000-8000-00805f9b34fb";
    public static final String UUID_SENSOR_CHAR = "25047e64-657c-4856-afcf-e315048a965b";
    public static final String UUID_BATTERY_CHAR = "00002a19-0000-1000-8000-00805f9b34fb";
    public static final byte[] CMD_START_SENSOR = {126, 3, 24, -42, 1, 0, 0};
    public static final byte[] CMD_STOP_SENSOR = {126, 3, 24, -42, 0, 0, 0};
    public static final byte[] CMD_SET_RTC_MS = {126, 10, 24, 30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    public static final int CONNECTION_LIMIT = 6;
    public static final String DEVICE_FILTER = "QM-SS1";
}