using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
using UnityEngine.UI;

public class main : MonoBehaviour {

    public GameObject gameObject;
    public Text text;

    string ipAddr = "192.168.116.73"; 
    string ipAddr2 = "192.168.116.72";

    Vector3 vector3 = new Vector3(0,0,0);
    UDPSystem udpSystem;
    char device = 'A';

    private void Awake()
    {
        switch (device) {
            case 'A':
                udpSystem = new UDPSystem(null);
                udpSystem.Set(ipAddr, 5001, ipAddr2, 5002);
                break;
            case 'B':
                udpSystem = new UDPSystem((x) => Receive(x));
                udpSystem.Set(ipAddr2, 5002, ipAddr, 5001);
                udpSystem.Receive();
                break;
        }
    }

    // Use this for initialization
    void Start () {
		
	}
	
	// Update is called once per frame
	void Update () {
        if (device == 'A')
        {
            vector3 = gameObject.transform.position;
            DATA sendData = new DATA(vector3);
            udpSystem.Send(sendData.ToByte(),99);
        }
        if (device == 'B')
        {
            gameObject.transform.position = vector3;
        }
        text.text = "(" + vector3.x + "," + vector3.y + "," + vector3.z + ")";
    }

    void Receive(byte[] bytes)
    {
        DATA getData = new DATA(bytes);
        vector3 = getData.ToVector3();
    }
}
public class DATA
{
    private float x;
    private float y;
    private float z;

    public DATA(byte[] bytes)
    {
        x = BitConverter.ToSingle(bytes, 0);
        y = BitConverter.ToSingle(bytes, 4);
        z = BitConverter.ToSingle(bytes, 8);
    }

    public DATA(Vector3 vector3){
        x = vector3.x;
        y = vector3.y;
        z = vector3.z;
    }

    public byte[] ToByte()
    {
        byte[] x = BitConverter.GetBytes(this.x);
        byte[] y = BitConverter.GetBytes(this.y);
        byte[] z = BitConverter.GetBytes(this.z);
        return MargeByte(MargeByte(x, y), z);
    }

    public Vector3 ToVector3()
    {
        return new Vector3(this.x, this.y, this.z);
    }

    public static byte[] MargeByte(byte[] baseByte,byte[] addByte)
    {
        byte[] b = new byte[baseByte.Length + addByte.Length];
        for(int i = 0; i < b.Length; i++)
        {
            if (i < baseByte.Length) b[i] = baseByte[i];
            else b[i] = addByte[i - baseByte.Length];
        }
        return b;
    }

    public static byte[] CutByte(byte[] baseByte, int startIndex, int size)
    {
        byte[] b = new byte[size];
        for (int i = 0; i < size; i++)
        {
            b[i] = baseByte[startIndex + i];
        }
        return b;
    }
}