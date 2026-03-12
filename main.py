import sys
import cv2
from datetime import datetime
import time
import serial
import os
import serial.tools.list_ports
from PyQt6.QtCore import QThread, pyqtSignal, Qt

from PyQt6.QtWidgets import (
    QApplication, QLabel, QWidget,
    QVBoxLayout, QHBoxLayout, QFrame
)
from PyQt6.QtGui import QImage, QPixmap, QFont
from PyQt6.QtCore import Qt
from face_engine import FaceEngine

class SerialThread(QThread):

    data_signal = pyqtSignal(str)

    def __init__(self, port="/dev/ttyUSB0", baud=115200):
        super().__init__()
        self.port = port
        self.baud = baud
        self.running = True
        port1 = self.find_esp32_port()
        if port1:
            self.port = port1

    def run(self):

        try:
            ser = serial.Serial(self.port, self.baud, timeout=1)
            print("Serial opened:", self.port)

        except Exception as e:
            print("Không mở được Serial:", e)
            return

        while self.running:
            if ser.in_waiting:
                line = ser.readline().decode().strip()
                self.data_signal.emit(line)
                time.sleep(0.01)
        ser.close()
    def find_esp32_port(self):
        ports = serial.tools.list_ports.comports()

        for port in ports:
            if "USB" in port.description or "CP210" in port.description or "CH340" in port.description:
                print("ESP32 found:", port.device)
                return port.device

        return None

    def stop(self):
        self.running = False
        self.quit()
        self.wait()
# ===== Thread camera + recognition =====
class CameraThread(QThread):
    frame_signal = pyqtSignal(object, str, float)   # frame, name, fps

    def __init__(self):
        super().__init__()
        self.engine = FaceEngine()
        self.running = True
        self.enable_recognition = True
        self.prev_time = time.time()
    def find_usb_camera(self):
        for dev in os.listdir("/dev"):
            if dev.startswith("video"):
                path = f"/dev/{dev}"
                cap = cv2.VideoCapture(path)
                if cap.isOpened():
                    print("Camera found:", path)
                    cap.release()
                    return path
        return None

    def run(self):
        cam_index = self.find_usb_camera()
        if cam_index:
            print("Using camera:", cam_index)
            cap = cv2.VideoCapture(cam_index)
        else:
            cap = cv2.VideoCapture("/dev/video0")

        if not cap.isOpened():
            print("Không mở được camera")
            return

        while self.running:
            ret, frame = cap.read()
            now = time.time()
            fps = int(1 / max(now - self.prev_time, 0.001))
            self.prev_time = now
            if ret:
                frame = cv2.resize(frame, (640, 480))
                start_time = time.time()
                if self.enable_recognition:
                    frame, name = self.engine.process_frame(frame)
                else:
                    name = ""

                self.frame_signal.emit(frame, name, fps)

        cap.release()

    def stop(self):
        self.running = False
        self.quit()
        self.wait()


# ===== GUI =====
class App(QWidget):
    def __init__(self):
        super().__init__()

        self.current_name = None
        self.current_detect_time = 0
        self.hold_duration = 3
        self.mode = "A"

        self.rfid_wait = None
        self.rfid_time = 0
        self.rfid_window = 1
        self.serial_thread = SerialThread("/dev/ttyUSB0", 115200)
        self.serial_thread.data_signal.connect(self.handle_serial)
        self.serial_thread.start()

        self.setWindowTitle("Attendance System")
        self.resize(1100, 650)

        # ===== GLOBAL STYLE =====
        self.setStyleSheet("""
            QWidget {
                background-color: #1e1e2f;
                color: white;
            }
            QFrame {
                background-color: #2c2c3e;
                border-radius: 15px;
            }
        """)

        main_layout = QVBoxLayout()

        # ===== HEADER =====
        header = QLabel("ATTENDANCE SYSTEM")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.setFont(QFont("Arial", 28, QFont.Weight.Bold))
        header.setStyleSheet("color: #00ffcc; padding: 15px;")

        main_layout.addWidget(header)

        # ===== BODY LAYOUT =====
        body_layout = QHBoxLayout()

        # ================= CAMERA PANEL =================
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("""
            background-color: black;
            border-radius: 15px;
        """)
        self.image_label.setMinimumSize(700, 500)

        body_layout.addWidget(self.image_label, 2)

        # ================= INFO PANEL =================
        info_frame = QFrame()
        info_layout = QVBoxLayout()

        self.name_label = QLabel("👤 TÊN: Chưa nhận diện")
        self.time_label = QLabel("🕒 THỜI GIAN: --:--:--")
        self.status_label = QLabel("📌 TRẠNG THÁI: ĐANG CHỜ")
        self.fps_label = QLabel("⚡ FPS: 0")
        self.mode_label = QLabel("Chế độ điểm danh: --")
        self.mode_label.setFont(QFont("Arial", 28))
        self.mode_label.setStyleSheet("color: #00ffcc;")

        font_big = QFont("Arial", 28, QFont.Weight.Bold)

        self.name_label.setFont(font_big)
        self.time_label.setFont(font_big)
        self.status_label.setFont(font_big)

        self.fps_label.setFont(QFont("Arial", 14))

        info_layout.addStretch()
        info_layout.addWidget(self.name_label)
        info_layout.addWidget(self.time_label)
        info_layout.addWidget(self.status_label)
        info_layout.addSpacing(30)
        info_layout.addWidget(self.fps_label)
        info_layout.addStretch()

        info_frame.setLayout(info_layout)
        body_layout.addWidget(info_frame, 1)

        info_layout.addWidget(self.mode_label)

        main_layout.addLayout(body_layout)
        self.setLayout(main_layout)

        # ===== THREAD =====
        self.thread = CameraThread()
        self.thread.enable_recognition = False
        self.users = self.thread.engine.db.load_users()
        self.thread.frame_signal.connect(self.update_ui)
        self.thread.start()

    def update_ui(self, frame, name, fps):
        now = time.time()
        # ===== Hiển thị camera =====
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qt_image = QImage(
            rgb.data,
            w,
            h,
            ch * w,
            QImage.Format.Format_RGB888
        )

        # self.image_label.setPixmap(QPixmap.fromImage(qt_image))
        pixmap = QPixmap.fromImage(qt_image)

        scaled_pixmap = pixmap.scaled(
            self.image_label.width(),
            self.image_label.height(),
            Qt.AspectRatioMode.KeepAspectRatioByExpanding,
            Qt.TransformationMode.SmoothTransformation
        )

        self.image_label.setPixmap(scaled_pixmap)
        self.fps_label.setText(f"FPS: {fps}")

        now = time.time()

        # ===== Nếu detect hợp lệ =====
        if name and name != "Unknown":

            idcard = name
            user = self.users.get(idcard)

            if not user:
                return

            display_name = user["Name"]

            if self.mode == "B":

                if display_name != self.current_name:
                    self.current_name = display_name
                    self.current_detect_time = now
                else:
                    self.current_detect_time = now

            elif self.mode == "A":

                if self.rfid_wait and (now - self.rfid_time <= self.rfid_window):

                    if user["RFID"] == self.rfid_wait:

                        self.current_name = display_name
                        self.current_detect_time = now
                        print("Attendance success")

                    else:
                        print("RFID mismatch")

                    self.rfid_wait = None
                    self.thread.enable_recognition = False

            if self.mode == "A":

                if self.rfid_wait and (now - self.rfid_time > self.rfid_window):

                    self.rfid_wait = None
                    self.thread.enable_recognition = False
        # reset sau 3 giây
        if self.current_name and (now - self.current_detect_time > self.hold_duration):
            self.current_name = None
        # ===== Update UI text =====
        if self.current_name:
            self.name_label.setText(f"TÊN: {self.current_name}")
            self.time_label.setText(
                f"THỜI GIAN: {datetime.now().strftime('%H:%M:%S')}"
            )
            self.status_label.setText("TRẠNG THÁI: ĐÃ NHẬN DIỆN")
            self.status_label.setStyleSheet("color: green")
        else:
            self.name_label.setText("TÊN: Chưa nhận diện")
            self.time_label.setText("THỜI GIAN: --:--:--")
            self.status_label.setText("TRẠNG THÁI: ĐANG CHỜ")
            self.status_label.setStyleSheet("color: black")
        # ===== Hiển thị MODE =====
        if self.mode == "A":
            self.mode_label.setText("Chế độ điểm danh: Thẻ từ")

        elif self.mode == "B":
            self.mode_label.setText("Chế độ điểm danh: Hàng loạt")

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.close()

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()
    def handle_serial(self, data):
        data = data.strip()
        print("Serial:", data)

        if data == "ModeA":
            self.mode = "A"
            self.thread.enable_recognition = False
            print("MODE A")

        elif data == "ModeB":
            self.mode = "B"
            self.thread.enable_recognition = True
            print("MODE B")

        elif data.startswith("rfid:"):

            rfid = data.split(":")[1]

            self.rfid_wait = rfid
            self.rfid_time = time.time()

            self.thread.enable_recognition = True


# ===== Main =====
app = QApplication(sys.argv)
window = App()
# window.show()
window.showFullScreen()
sys.exit(app.exec())