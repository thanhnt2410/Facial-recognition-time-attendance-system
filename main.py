import sys
import cv2
from datetime import datetime
import time

from PyQt5.QtWidgets import (
    QApplication, QLabel, QWidget, QVBoxLayout
)
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import QThread, pyqtSignal, Qt

from face_engine import FaceEngine


# ===== Thread camera + recognition =====
class CameraThread(QThread):
    frame_signal = pyqtSignal(object, str, float)   # thêm fps

    def __init__(self):
        super().__init__()
        self.engine = FaceEngine()

    def run(self):
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Không mở được camera")
            return

        prev_time = time.time()

        while True:
            ret, frame = cap.read()
            if ret:
                start_time = time.time()

                frame, name = self.engine.process_frame(frame)

                # ===== FPS calculation =====
                end_time = time.time()
                fps = 1 / (end_time - start_time)

                self.frame_signal.emit(frame, name, fps)


# ===== GUI =====
class App(QWidget):
    def __init__(self):
        super().__init__()
        self.last_attendance_time = {}
        self.cooldown = 3  # giây
        self.current_name = None
        self.current_detect_time = 0
        self.hold_duration = 3  # giữ thông tin 3 giây

        self.setWindowTitle("Attendance System")

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)

        font = QFont("Arial", 16)

        self.name_label = QLabel("TÊN: Chưa nhận diện")
        self.time_label = QLabel("THỜI GIAN: --:--:--")
        self.status_label = QLabel("TRẠNG THÁI: ĐANG CHỜ")
        self.fps_label = QLabel("FPS: 0")   # thêm dòng này

        self.name_label.setFont(font)
        self.time_label.setFont(font)
        self.status_label.setFont(font)
        self.fps_label.setFont(font)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.name_label)
        layout.addWidget(self.time_label)
        layout.addWidget(self.status_label)
        layout.addWidget(self.fps_label)   # thêm vào layout

        self.setLayout(layout)

        self.thread = CameraThread()
        self.thread.frame_signal.connect(self.update_ui)
        self.thread.start()

    # def update_ui(self, frame, name, fps):

    #     # ===== Hiển thị camera =====
    #     rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #     h, w, ch = rgb.shape
    #     qt_image = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
    #     self.image_label.setPixmap(QPixmap.fromImage(qt_image))

    #     self.fps_label.setText(f"FPS: {fps:.2f}")

    #     current_time = time.time()

    #     # ===== Nếu không nhận diện được =====
    #     if not name or name == "Unknown":
    #         self.name_label.setText("TÊN: Chưa nhận diện")
    #         self.time_label.setText("THỜI GIAN: --:--:--")
    #         self.status_label.setText("TRẠNG THÁI: ĐANG CHỜ")
    #         self.status_label.setStyleSheet("color: black")
    #         return

    #     # ===== Có nhận diện =====
    #     if name in self.last_attendance_time:
    #         if current_time - self.last_attendance_time[name] < self.cooldown:
    #             self.status_label.setText("TRẠNG THÁI: ĐÃ ĐIỂM DANH")
    #             self.status_label.setStyleSheet("color: orange")
    #             return

    #     # ===== Ghi nhận điểm danh =====
    #     self.last_attendance_time[name] = current_time

    #     self.name_label.setText(f"TÊN: {name}")
    #     self.time_label.setText(
    #         f"THỜI GIAN: {datetime.now().strftime('%H:%M:%S')}"
    #     )
    #     self.status_label.setText("TRẠNG THÁI: NHẬN DIỆN THÀNH CÔNG")
    #     self.status_label.setStyleSheet("color: green")
    def update_ui(self, frame, name, fps):

        # ===== Hiển thị camera =====
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qt_image = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qt_image))

        self.fps_label.setText(f"FPS: {fps:.2f}")

        now = time.time()

        # ===== Nếu nhận diện được người hợp lệ =====
        if name and name != "Unknown":

            # Nếu là người mới
            if name != self.current_name:
                self.current_name = name
                self.current_detect_time = now

            else:
                # Cập nhật lại thời gian nếu vẫn là cùng người
                self.current_detect_time = now

        # ===== Nếu không detect được =====
        else:
            # Nếu đã quá thời gian giữ
            if self.current_name and (now - self.current_detect_time > self.hold_duration):
                self.current_name = None

        # ===== Hiển thị UI =====
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
        if event.key() == Qt.Key_Escape:
            self.close()   # thoát app


app = QApplication(sys.argv)
window = App()
window.show()
# window.showFullScreen()
sys.exit(app.exec_())