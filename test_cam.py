import os
import cv2
import serial.tools.list_ports

def find_usb_camera():
    for dev in os.listdir("/dev"):
        if dev.startswith("video") and dev != "video0":   # bỏ webcam laptop
            path = f"/dev/{dev}"
            cap = cv2.VideoCapture(path)
            if cap.isOpened():
                print("Camera found:", path)
                cap.release()
                return path
    return None


def find_esp32_port():
    ports = serial.tools.list_ports.comports()

    for port in ports:
        if "USB" in port.description or "CP210" in port.description or "CH340" in port.description:
            print("ESP32 found:", port.device)
            return port.device

    return None

cam_index = find_usb_camera()
# port = find_esp32_port()
# print(port)