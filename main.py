
import sys
import cv2
from datetime import datetime
import time
from PyQt6.QtCore import QThread, pyqtSignal, Qt

from PyQt6.QtWidgets import (
    QApplication, QLabel, QWidget,
    QVBoxLayout, QHBoxLayout, QFrame
)
from PyQt6.QtGui import QImage, QPixmap, QFont
from PyQt6.QtCore import Qt
from face_engine import FaceEngine


# ===== Thread camera + recognition =====
class CameraThread(QThread):
    frame_signal = pyqtSignal(object, str, float)   # frame, name, fps

    def __init__(self):
        super().__init__()
        self.engine = FaceEngine()
        self.running = True

    def run(self):
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Không mở được camera")
            return

        while self.running:
            ret, frame = cap.read()
            if ret:
                start_time = time.time()

                frame, name = self.engine.process_frame(frame)

                end_time = time.time()
                fps = 1 / (end_time - start_time)

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

        main_layout.addLayout(body_layout)
        self.setLayout(main_layout)

        # ===== THREAD =====
        self.thread = CameraThread()
        self.thread.frame_signal.connect(self.update_ui)
        self.thread.start()

    def update_ui(self, frame, name, fps):

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
        self.fps_label.setText(f"FPS: {fps:.2f}")

        now = time.time()

        # ===== Nếu detect hợp lệ =====
        if name and name != "Unknown":

            if name != self.current_name:
                self.current_name = name
                self.current_detect_time = now
            else:
                self.current_detect_time = now

        else:
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

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.close()

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()


# ===== Main =====
app = QApplication(sys.argv)
window = App()
# window.show()
window.showFullScreen()
sys.exit(app.exec())