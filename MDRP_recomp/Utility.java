package com.sony.mocopi.datareader.ble;

import android.util.Half;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/* loaded from: classes.jar:com/sony/mocopi/datareader/ble/Utility.class */
public class Utility {
    public static long convertByteArrayToLong(byte[] bytes, int offset) {
        byte[] longBytes = new byte[8];
        System.arraycopy(bytes, offset, longBytes, 0, 8);
        ByteBuffer buffer = ByteBuffer.wrap(longBytes).order(ByteOrder.LITTLE_ENDIAN);
        return buffer.getLong();
    }

    public static short getInt16(byte[] bytes, int offset, ByteOrder order) {
        ByteBuffer buffer = ByteBuffer.wrap(bytes).order(order);
        return buffer.getShort(offset);
    }

    public static float getFloat16(byte[] bytes, int offset) {
        short halfFloatBits = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).getShort(offset);
        return Half.valueOf(halfFloatBits).floatValue();
    }
}