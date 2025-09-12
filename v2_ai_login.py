import os
import sys
import csv
import threading
import psutil
import time
import subprocess
import numpy as np
import torch
import yaml
sys.path.insert(0, './yolov5')
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QGridLayout, QPushButton,
    QHBoxLayout, QVBoxLayout, QDialog, QRadioButton, QDialogButtonBox,
    QCheckBox, QScrollArea, QDesktopWidget, QMessageBox, QSlider, QMenu, QAction
)
from PyQt5.QtGui import QPixmap, QImage, QFont, QPainter, QPen
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QSize, QTimer, QRect
from PyQt5 import QtWidgets
from models.experimental import attempt_load
from utils.general import non_max_suppression, LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2, increment_path, scale_boxes, strip_optimizer, xyxy2xywh
from utils.torch_utils import select_device, smart_inference_mode
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from models.common import DetectMultiBackend
from ultralytics.utils.plotting import Annotator, colors, save_one_box
from utils.plots import colors
from PyQt5 import QtGui, QtCore
import qdarkstyle
import cv2
import pygame
import json
from collections import deque
from mss import mss
roi_file = "rois.json"
pygame.mixer.init()
from PIL import Image
g_device = "cuda" if torch.cuda.is_available() else "cpu"
class AlarmHandler:
    def __init__(self):
        self.alarm_active = False
        self.cooldown_active = False
        self.grey_zone_active = False

        # increasing the following time would reduce the alarm senstivity
        # subsequently increasing the risk of missed alarms

        self.grey_zone_time = 0
        self.grey_zone_timer = QTimer()
        self.grey_zone_timer.setSingleShot(True)
        self.grey_zone_timer.timeout.connect(self.grey_zone_trigger)
        self.black_zone_active = False

        # increasing the following would extend the window of confidence i.e. alarm can played once grey zone is over.
        self.black_zone_time = 25000
        self.black_zone_timer = QTimer()
        self.black_zone_timer.setSingleShot(True)
        self.black_zone_timer.timeout.connect(self.black_zone_trigger)

        self.cooldown_timer = QTimer()
        self.cooldown_timer.setSingleShot(True)
        self.cooldown_timer.timeout.connect(self.end_cooldown)

        self.channel = pygame.mixer.Channel(0)
        self.sound = [pygame.mixer.Sound("alarm0.mp3"), pygame.mixer.Sound("alarm1.mp3"), pygame.mixer.Sound("alarm2.mp3")]
        self.current_p = None

    def grey_zone_trigger(self):
        self.grey_zone_active = False
        self.black_zone_active = True
        self.black_zone_timer.start(self.black_zone_time)

    def black_zone_trigger(self):
        self.black_zone_active = False

    def grey_zone_start(self):
        self.grey_zone_active = True
        self.grey_zone_timer.start(self.grey_zone_time)
    
    def trigger_alarm(self,  p):
        if not self.black_zone_active and not self.grey_zone_active:
            self.grey_zone_start()

        elif self.current_p is not None and p > self.current_p and self.black_zone_active:
            self.current_p = p
            self.channel.stop()
            self.alarm_active = True
            self.black_zone_timer.stop()
            self.black_zone_timer.setInterval(self.black_zone_time)
            self.black_zone_timer.start()
            self.black_zone_active = True
            self.channel.play(self.sound[p])
            self.cooldown_active = True
            self.cooldown_timer.start(1000)
            self.alarm_active = False

        elif self.black_zone_active and not self.alarm_active and not self.cooldown_active:
            self.current_p = p
            self.alarm_active = True
            self.black_zone_timer.stop()
            self.black_zone_timer.setInterval(self.black_zone_time)
            self.black_zone_timer.start()            
            self.black_zone_active = True
            self.channel.play(self.sound[p])
            self.cooldown_active = True
            self.cooldown_timer.start(1000)
            self.alarm_active = False

    def end_cooldown(self):
        self.cooldown_active = False

    def alarm_finished(self):
        self.alarm_active = False

class WelcomePage(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Welcome to Smart Sentry")
        # self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint)  # Remove window frame
        self.setStyleSheet("""
            QDialog {
                border-image: url('ai.webp') 0 0 0 0 stretch stretch;
            }
        """)
        # Logo
        self.logo_label = QLabel()
        pixmap = QPixmap('logo4.png')
        self.logo_label.setPixmap(pixmap)
        self.logo_label.setAlignment(Qt.AlignCenter)
        self.logo_label.setStyleSheet("background-color: rgba(0, 0, 0, 0);")
        # Make logo glow on hover
        # Start button
        self.start_button = QPushButton("Launching System! Please Wait")
        self.start_button.setStyleSheet("""
            QPushButton {
                background-color: #D32F2F; /* Green */
                border: none;
                color: white;
                padding: 15px 32px;
                text-align: center;
                text-decoration: none;
                display: inline-block;
                font-size: 28px;
                margin: 4px 2px;
                cursor: pointer;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #C62828;
                border: 2px solid yellow;
            }
        """)
        self.start_button.setEnabled(False)
        self.start_button.clicked.connect(self.accept)

        # Layout
        main_layout = QVBoxLayout()
        main_layout.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.logo_label)
        main_layout.addWidget(self.start_button)
        self.setLayout(main_layout)

        # Set geometry to fullscreen
        self.setGeometry(QApplication.desktop().availableGeometry())

    def ready(self):
        self.start_button.setText("Press to Start!")
        self.start_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50; /* Green */
                border: none;
                color: white;
                padding: 15px 32px;
                text-align: center;
                text-decoration: none;
                display: inline-block;
                font-size: 28px;
                margin: 4px 2px;
                cursor: pointer;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #45a049;
                border: 2px solid yellow;
            }
        """)
        self.start_button.setEnabled(True)
        QtWidgets.QApplication.processEvents()

    def center(self):
        # Function to center the dialog on the screen
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

# class AsyncRead(threading.Thread):
#     def __init__(self, capture, name='AsyncRead'):
#         self.capture = capture
#         assert self.capture.isOpened()
#         self.cond = threading.Condition()
#         self.running = False
#         self.frame = None
#         self.latestnum = 0
#         self.buffer_size = 1  # Number of frames to buffer
#         self.frame_buffer = deque()
#         self.callback = None
#         self.fps = 0.0
#         super().__init__(name=name)
#         self.start_time = time.time()
#         self.start()

#     def start(self):
#         self.running = True
#         super().start()

#     def release(self, timeout=None):
#         self.running = False
#         self.join(timeout=timeout)
#         self.capture.release()

#     def run(self):
#         counter = 0
#         while self.running:
#             (rv, img) = self.capture.read()
#             if not rv:
#                 continue

#             counter += 1

#             # Add frame to buffer
#             with self.cond:
#                 self.frame_buffer.append(img)

#                 # If buffer exceeds specified size, remove the oldest frame
#                 if len(self.frame_buffer) > self.buffer_size:
#                     self.frame_buffer.popleft()

#                 # Calculate FPS
#                 # Combine frames from the buffer for smoother display
#                 self.frame = img
#                 self.latestnum = counter
#                 self.cond.notify_all()

#             if self.callback:
#                 self.callback(img)

#     def isOpened(self):
#         return self.capture.isOpened()

#     def read(self, wait=True, seqnumber=None, timeout=None):
#         with self.cond:
#             if wait:
#                 if (len(self.frame_buffer) > 0):
#                     to_ret = (1 , self.frame_buffer[0])
#                     self.frame_buffer.popleft()
#                     return to_ret
#                 if seqnumber is None:
#                     seqnumber = self.latestnum + 1
#                 if seqnumber < 1:
#                     seqnumber = 1

#                 rv = self.cond.wait_for(lambda: self.latestnum >= seqnumber, timeout=timeout)
#                 if not rv:
#                     return (self.latestnum, self.frame)

#             return (self.latestnum, self.frame)
class AsyncRead(threading.Thread):
	def __init__(self, capture, name='FreshestFrame'):
		self.capture = capture
		self.cond = threading.Condition()
		self.running = False
		self.frame = None
		self.latestnum = 0	
		self.callback = None
		super().__init__(name=name)
		self.start()

	def start(self):
		self.running = True
		super().start()

	def release(self, timeout=None):
		self.running = False
		self.join(timeout=timeout)
		self.capture.release()

	def run(self):
		counter = 0
		while self.running:
			(rv, img) = self.capture.read()
			# assert rv
			counter += 1
			with self.cond:
				self.frame = img if rv else None
				self.latestnum = counter
				self.cond.notify_all()

			if self.callback:
				self.callback(img)

	def read(self, wait=True, seqnumber=None, timeout=None):
		with self.cond:
			if wait:
				if seqnumber is None:
					seqnumber = self.latestnum+1
				if seqnumber < 1:
					seqnumber = 1
				
				rv = self.cond.wait_for(lambda: self.latestnum >= seqnumber, timeout=timeout)
				if not rv:
					return (self.latestnum, self.frame)
			return (self.latestnum, self.frame)

class ClassSelectionDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Detection Classes")

        self.camera_checkboxes = []
        self.camera_info = ["Animal", "Person", "Vehicle"]

        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)

        checkbox_font = QFont()
        checkbox_font.setPointSize(12)  # Set a larger font size

        for info in self.camera_info:
            checkbox = QCheckBox(info)
            checkbox.setFont(checkbox_font)
            checkbox.setStyleSheet("""
                QCheckBox {
                    padding: 10px;
                    border: 2px solid black;
                    border-radius: 5px;
                }
                QCheckBox::indicator {
                    width: 20px;
                    height: 20px;
                    border-radius: 5px;
                }
                QCheckBox::indicator:checked {
                    background-color: green;
                    border: 2px solid darkgreen;
                }
                QCheckBox::indicator:unchecked {
                    background-color: white;
                    border: 2px solid black;
                }
                QCheckBox::indicator:checked:hover {
                    background-color: lightgreen;
                }
            """)
            checkbox.setIconSize(QSize(0, 0))  # Hide the default checkbox icon
            checkbox.setChecked(True)  # Default to selected
            scroll_layout.addWidget(checkbox)
            self.camera_checkboxes.append(checkbox)

        scroll_widget.setLayout(scroll_layout)
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)  # Allow scroll area to resize with content

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        main_layout = QVBoxLayout()
        main_layout.addWidget(scroll_area)
        main_layout.addWidget(button_box)

        self.setLayout(main_layout)

        # Set dialog size
        self.resize(700, 400)  # Example dimensions

        # Center the dialog on the screen
        self.center()

    def get_selected_classes(self):
        selected_classes = set()
        if self.camera_checkboxes[0].isChecked():
            selected_classes.add("Animal")
        if self.camera_checkboxes[1].isChecked():
            selected_classes.add("Person")
        if self.camera_checkboxes[2].isChecked():
            selected_classes.add("Vehicle")

        return selected_classes

    def center(self):
        # Function to center the dialog on the screen
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

class CaptureIpCameraFramesWorker(QThread):
    ImageUpdated = pyqtSignal(QImage)
    HighlightCamera = pyqtSignal(bool)  # New signal for highlighting camera label
    toggleStateChanged = pyqtSignal(bool)
    AlarmTriggered = pyqtSignal(int)
    def __init__(self, url, selected_classes, threshold, parent=None):
        super().__init__(parent)
        self.url = url
        self.__thread_active = True
        self.__thread_pause = False
        self.cap = None
        self.priority = 0
        self.screen = None
        self.screen_mode = self.urls_screen()
        self.frame_size = QSize(640, 480)  # Default size
        self.selected_classes = {}
        self.model  = torch.hub.load('./yolov5', 'custom',source='local', path='yolov5s.pt',device=g_device, force_reload=True)
        self.model.conf=threshold
        self.model.eval()
        if g_device == "cuda":
            self.model.cuda()
            self.model.to('cuda')
        self.classes = {}
        self.time = []
        self.poly = None
        self.ers = None
        self.worker_active = True  # Worker starts as active
        self.toggle_state = True  # Initial toggle state
        self.frames_per_detection = 3
        self.detect_now = 0
        if 'Person' in selected_classes:
            self.classes.update(dict.fromkeys([0], 'Person'))
        if 'Vehicle' in selected_classes:
            self.classes.update(dict.fromkeys([1, 2, 3, 5, 7], 'Vehicle'))
        if 'Animal' in selected_classes:
            self.classes.update(dict.fromkeys([14, 15, 16, 17, 18, 19], 'Animal'))
        self.safe_roi = threading.Condition()
        self.safe_p = threading.Condition()
        self.show_ers = False
        self.state_stream = False

    def urls_screen(self):
        if len(self.url) != 2:
            return False
        if self.url[0] == "s" and self.url[1].isnumeric():
            self.screen = int(self.url[1])
            return True
        else:
            return False


        
    def detect(self, frame):
        self.toggleStateChanged.emit(self.toggle_state)
        results = self.model(frame)
        condition_if = False
        draw_condition = False

        with self.safe_roi:
            if self.poly is not None:
                cv2.polylines(frame, self.poly, isClosed=True, color=(255, 0, 0), thickness=2)
            if self.show_ers and (self.ers is not None):
                for er in self.ers:
                    cv2.polylines(frame, np.array([er]), isClosed=True, color=(255, 255, 0), thickness=2)
            self.safe_roi.notify_all()
        d_class = None

        for detection in results.xyxy[0]:
            x1, y1, x2, y2, conf, d_class = detection
            c = int(d_class)
            if c in self.classes:
                with self.safe_roi:
                    draw_condition = self.any_one_rectangle_corners_inside_polygon(x1, y1, x2, y2, None if self.poly is None else self.poly[0]) and not self.any_ER(x1, y1, x2, y2)
                    condition_if = condition_if or draw_condition
                    self.safe_roi.notify_all()
                if draw_condition:
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, self.classes[c], (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        if condition_if:
            if self.toggle_state:
                self.HighlightCamera.emit(True)
                with self.safe_p:
                    self.AlarmTriggered.emit(self.priority)
                    self.safe_p.notify_all()
        else: 
            if not self.toggle_state:
                pass
            else:
                self.HighlightCamera.emit(False)
                
        return frame
    
    def any_ER(self, x1, y1, x2, y2):
        comp = False
        if self.ers is not None:
            for er in self.ers:
                comp = comp or self.any_one_rectangle_corners_inside_polygon(x1, y1, x2, y2, er)
        return comp

        
    def run(self):
        self.state_stream = False
        print("attempting restart!")
        if self.screen_mode:
            try:
                with mss() as sct:
                    monitor = sct.monitors[self.screen]  # Target monitor
                    while self.__thread_active:
                        screenshot = sct.grab(monitor)
                        frame = np.array(screenshot, dtype=np.uint8)
                        height, width, channels = frame.shape
                        bytes_per_line = width * channels
                        # cv_rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        cv_rgb_image = self.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        qt_image = QImage(cv_rgb_image.data, cv_rgb_image.shape[1], cv_rgb_image.shape[0], cv_rgb_image.shape[1]*3, QImage.Format_RGB888)
                        self.ImageUpdated.emit(qt_image)
            except:
                print(f"Monitor: {self.screen} is not available!")
                img = np.array(Image.open("no_signal.png"))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                h, w, c = img.shape
                bytes_per = w * c
                qt_rgb_ = QImage(img.data, w, h, bytes_per, QImage.Format_RGB888)
                self.ImageUpdated.emit(qt_rgb_)
                self.stop()
                return
        else:
            self.cap = cv2.VideoCapture(self.url)
            if not self.cap.isOpened():
                print(f"Error: Could not open camera {self.url}")
                img = np.array(Image.open("no_signal.png"))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                h, w, c = img.shape
                bytes_per = w * c
                qt_rgb_ = QImage(img.data, w, h, bytes_per, QImage.Format_RGB888)
                self.ImageUpdated.emit(qt_rgb_)
                self.cap.release()
                self.release()
                self.stop()
                return
            self.cap = AsyncRead(self.cap)
            self.state_stream = True
            while self.__thread_active:
                if not self.__thread_pause:
                    ret, frame = self.cap.read(True)
                    if frame is not None and ret:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        height, width, channels = frame.shape
                        bytes_per_line = width * channels
                        cv_rgb_image = self.detect(frame)
                        qt_image = QImage(cv_rgb_image.data, cv_rgb_image.shape[1], cv_rgb_image.shape[0], cv_rgb_image.shape[1]*3, QImage.Format_RGB888)
                        self.ImageUpdated.emit(qt_image)
                    else:
                        break
            self.state_stream = False
            print("stream exit occurred!")
            self.stop()

    def any_one_rectangle_corners_inside_polygon(self, x1, y1, x2, y2, polygon_corners):

        # Check all pairs of rectangle corners
        if polygon_corners is None:
            return True
        count_inside = 0
        for corner in [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]:
            if self.point_in_polygon(corner, polygon_corners):
                count_inside += 1
                if count_inside == 2:
                    return True

        return False


    
    def point_in_polygon(self, point, polygon_corners):
        # Ensure polygon_corners is converted to a NumPy array

        if len(polygon_corners) == 0:
            return True  # Handle case where polygon_corners is empty
        x, y = point
        x_coords, y_coords = polygon_corners[:, 0], polygon_corners[:, 1]

        # Check if the point is inside the bounding box defined by the polygon corners
        if np.min(x_coords) <= x <= np.max(x_coords) and np.min(y_coords) <= y <= np.max(y_coords):
            # Point is potentially inside the bounding box, proceed to more detailed check
            num_vertices = len(polygon_corners)
            inside = False
            j = num_vertices - 1
            for i in range(num_vertices):
                if (y_coords[i] > y) != (y_coords[j] > y):
                    if x < (x_coords[j] - x_coords[i]) * (y - y_coords[i]) / (y_coords[j] - y_coords[i]) + x_coords[i]:
                        inside = not inside
                j = i
            return inside
        else:
            return False


    def stop(self):
        if self.cap:
            self.cap.release()
            self.cap = None
            self.state_stream = False

    def pause(self):
        self.__thread_pause = True

    def unpause(self):
        self.__thread_pause = False

    def update_frame_size(self, size):
        self.frame_size = size

    def toggleState(self):
        self.toggle_state = not self.toggle_state
        # Optionally, you can emit a signal here to notify the UI about the state change
        self.toggleStateChanged.emit(self.toggle_state)
        self.worker_active = not self.worker_active

    def isToggleOn(self):
        return self.toggle_state
    
    def release(self):
        self.cap.release()

class CameraSelectionDialog(QDialog):
    def __init__(self, camera_info, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Cameras")

        self.camera_info = camera_info
        self.camera_checkboxes = []

        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)

        checkbox_font = QFont()
        checkbox_font.setPointSize(12)  # Set a larger font size

        for info in self.camera_info:
            checkbox = QCheckBox(info['display_ips'])
            checkbox.setFont(checkbox_font)
            checkbox.setStyleSheet("""
                QCheckBox {
                    padding: 10px;
                    border: 2px solid black;
                    border-radius: 5px;
                }
                QCheckBox::indicator {
                    width: 20px;
                    height: 20px;
                    border-radius: 5px;
                }
                QCheckBox::indicator:checked {
                    background-color: green;
                    border: 2px solid darkgreen;
                }
                QCheckBox::indicator:unchecked {
                    background-color: white;
                    border: 2px solid black;
                }
                QCheckBox::indicator:checked:hover {
                    background-color: lightgreen;
                }
            """)
            checkbox.setIconSize(QSize(0, 0))  # Hide the default checkbox icon
            checkbox.setChecked(True)  # Default to selected
            scroll_layout.addWidget(checkbox)
            self.camera_checkboxes.append(checkbox)

        scroll_widget.setLayout(scroll_layout)
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)  # Allow scroll area to resize with content

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        main_layout = QVBoxLayout()
        main_layout.addWidget(scroll_area)
        main_layout.addWidget(button_box)

        self.setLayout(main_layout)

        # Set dialog size
        self.resize(700, 500)  # Example dimensions

        # Center the dialog on the screen
        self.center()

    def center(self):
        # Function to center the dialog on the screen
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def get_selected_cameras(self):
        selected_cameras = []
        for idx, checkbox in enumerate(self.camera_checkboxes):
            if checkbox.isChecked():
                selected_cameras.append(self.camera_info[idx])
        return selected_cameras

class ViewDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Choose View Mode")

        self.radio_2x2 = QRadioButton("2x2 View")
        self.radio_3x3 = QRadioButton("3x3 View")
        self.radio_4x4 = QRadioButton("4x4 View")
        self.radio_4x4.setChecked(True)  # Default to 4x4 view
        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        layout = QVBoxLayout()
        layout.addWidget(self.radio_2x2)
        layout.addWidget(self.radio_3x3)
        layout.addWidget(self.radio_4x4)
        layout.addWidget(self.button_box)

        self.setLayout(layout)

        # Apply stylesheet for radio buttons
        self.setStyleSheet("""
            QRadioButton {
                padding: 10px;
                border: 2px solid black;
                border-radius: 5px;
            }
            QRadioButton::indicator {
                width: 20px;
                height: 20px;
                border-radius: 5px;
            }
            QRadioButton::indicator:checked {
                background-color: green;
                border: 2px solid darkgreen;
            }
            QRadioButton::indicator:unchecked {
                background-color: white;
                border: 2px solid black;
            }
            QRadioButton::indicator:checked:hover {
                background-color: lightgreen;
            }
        """)

        # Set font for radio buttons
        radio_font = QFont()
        radio_font.setPointSize(12)  # Set a larger font size
        self.radio_2x2.setFont(radio_font)
        self.radio_3x3.setFont(radio_font)
        self.radio_4x4.setFont(radio_font)

        # Set dialog size
        self.resize(500, 300)
        self.center()

    def center(self):
        # Function to center the dialog on the screen
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def getViewMode(self):
        if self.radio_2x2.isChecked():
            return '2x2'
        elif self.radio_3x3.isChecked():
            return '3x3'
        elif self.radio_4x4.isChecked():
            return '4x4'
        else:
            return None

class FullScreenViewer(QDialog):
    def __init__(self, camera_worker):
        super().__init__()
        self.setWindowTitle("Fullscreen View")
        self.setAttribute(Qt.WA_DeleteOnClose)  # Ensure the window is completely destroyed on close
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint)  # Remove window frame

        self.label = QLabel()
        self.label.setAlignment(Qt.AlignCenter)
        self.camera_worker = camera_worker
        self.camera_worker.ImageUpdated.connect(self.update_image)
        self.camera_worker.start()

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        self.setLayout(layout)

        # Adjust window size to match screen size
        screen_size = QDesktopWidget().screenGeometry()
        self.screen_width = screen_size.width()
        self.screen_height = screen_size.height()
        self.resize(self.screen_width, self.screen_height)

    def update_image(self, frame: QImage):
        pixmap = QPixmap.fromImage(frame)
        scaled_pixmap = pixmap.scaled(self.screen_width, self.screen_height - 100, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.label.setPixmap(scaled_pixmap)

    def closeEvent(self, event):
        self.camera_worker.stop()
        super().closeEvent(event)

class FloatSlider(QSlider):
    valueChangedFloat = pyqtSignal(float)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setOrientation(Qt.Horizontal)
        self.setRange(0, 100)  # Set range from 0 to 100 (corresponding to 0.0 to 1.0)
        self.setSingleStep(1)  # Set the step size
        self.setValue(45)  # Initial value at 0.5
        self.valueChanged.connect(self.emitFloatValue)

    def emitFloatValue(self, value):
        float_value = value / 100.0  # Convert integer value to float between 0.0 and 1.0
        self.valueChangedFloat.emit(float_value)

class FloatSliderDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Detection Threshold")

        layout = QVBoxLayout(self)

        self.slider = FloatSlider()
        self.slider.valueChangedFloat.connect(self.onSliderValueChangedFloat)
        layout.addWidget(self.slider)

        self.label = QLabel("Value: 0.45")
        layout.addWidget(self.label)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.resize(600, 100)
        self.center()

    def onSliderValueChangedFloat(self, value):
        self.label.setText(f"Value: {value:.2f}")

    def getValue(self):
        return self.slider.value() / 100.0  # Return the current float value

    def center(self):
        # Function to center the dialog on the screen
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())
class ImageLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.points = []  # List to store polygon points as QPoint objects
        self.close_threshold = 10  # Threshold distance to consider closing the polygon
        self.accept_timer = QTimer(self)
        self.accept_timer.setSingleShot(True)
        self.accept_timer.timeout.connect(self.accept_roi)

    def clear_points(self):
        self.points = []
        self.update()

    def set_points(self, points):
        self.points = points
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(Qt.red, 2, Qt.SolidLine))

        if len(self.points) > 1:
            # Draw lines between consecutive points
            for i in range(len(self.points) - 1):
                painter.drawLine(self.points[i], self.points[i + 1])

            # Draw final line closing the polygon if it's closed
            if len(self.points) > 2:
                painter.drawLine(self.points[-1], self.points[0])

        # Draw points as clickable dots
        for point in self.points:
            painter.drawEllipse(point, 3, 3)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            if len(self.points) < 3:
                # Initial points selection
                self.points.append(event.pos())
                self.update()
            elif len(self.points) == 3:
                # Add fourth point and start timer
                self.points.append(event.pos())
                self.update()
                self.accept_timer.start(1000)  # Start timer for 1 second
            else:
                # Add new point to the polygon
                self.points[-1] = event.pos()
                self.update()

    def accept_roi(self):
        self.parent().show_roi_coordinates()  # Show ROI coordinates
        self.accept_timer.stop()  # Stop the timer

class ImageROIDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.image_label = ImageLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)

        self.setLayout(layout)
        self.setWindowTitle('Polygon ROI Selection')

        self.roi_coordinates = None  # Variable to store ROI coordinates
        self.original_frame_size = None  # Store original frame size

    def set_image_from_cv2(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channel = frame.shape

        self.original_frame_size = (width, height)  # Store original frame dimensions

        # Define max dimensions for the dialog window
        max_width = QApplication.desktop().screenGeometry().width() * 0.8
        max_height = QApplication.desktop().screenGeometry().height() * 0.8

        # Calculate scaling factors to fit within max dimensions
        scale_width = max_width / width
        scale_height = max_height / height
        scale_factor = min(scale_width, scale_height)

        # Resize frame to fit within max dimensions
        resized_width = int(width * scale_factor)
        resized_height = int(height * scale_factor)
        qimage = QImage(frame_rgb.data, width, height, width * channel, QImage.Format_RGB888)

        # Resize image label to fit the resized frame
        self.image_label.setPixmap(QPixmap.fromImage(qimage).scaled(resized_width, resized_height, Qt.KeepAspectRatio))

        # Resize dialog to fit the resized frame
        self.resize(resized_width, resized_height)

    def keyPressEvent(self, event):
        # Close the dialog and print coordinates on Enter key press
        if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            self.show_roi_coordinates()
        else:
            super().keyPressEvent(event)

    def show_roi_coordinates(self):
        if len(self.image_label.points) >= 3:
            # Extract coordinates and transform to original frame size
            roi_coordinates = np.array([[point.x() * self.original_frame_size[0] // self.image_label.width(),
                                         point.y() * self.original_frame_size[1] // self.image_label.height()]
                                        for point in self.image_label.points])

            self.roi_coordinates = roi_coordinates.astype(int)  # Convert to integers for pixel indices
            print(f"ROI Coordinates: {roi_coordinates}")

            # Close the dialog after printing coordinates
            self.accept()

    def get_roi_coordinates(self):
        return [self.roi_coordinates]

class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)



class MainWindow(QMainWindow):
    def __init__(self, state, initial_view_mode='4x4'):
        super(MainWindow, self).__init__()
        self.state_admin = state
        self.fn = 0
        self.welcome_page = WelcomePage(self)
        self.welcome_page.show()
        self.welcome_page.accepted.connect(self.start_application)
        self.camera_info = []  # List to hold camera information from CSV
        self.camera_workers = []
        self.camera_labels = []  # List to hold QLabel widgets
        self.current_window_index = 0
        self.num_windows = 0  # Calculate number of windows needed dynamically
        self.current_view_mode = initial_view_mode  # Set initial view mode
        screen_size = QDesktopWidget().screenGeometry()
        screen_width = screen_size.width()
        screen_height = screen_size.height()
        self.resize(screen_width, screen_height)
        self.master_btn = True
        self.load_camera_info_from_csv()
        self.setup_window()
        self.welcome_page.exec_()
        self.alarm_state = True
        self.center()

    def start_application(self):
        self.welcome_page.close()
        self.show()

    def center(self):
        # Function to center the dialog on the screen
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def load_camera_info_from_csv(self):
        try:
            with open('sources.csv', newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    if float(row['threshold']) > 1 or float(row['threshold']) < 0:
                        row['threshold'] = 0.5
                    self.camera_info.append({
                        'threshold' : row['threshold'],
                        'display_ips': row['display_ips'],
                        'ips': row['ips']  # Corrected 'ips' field name
                    })
        except FileNotFoundError:
            QMessageBox.critical(self, "File Error", "sources.csv not found or could not be opened.")
            sys.exit(1)

        self.num_windows = (len(self.camera_info) + 3) // 4  # Calculate number of windows needed
    
    def show_threshold_slider(self):
        dialog = FloatSliderDialog()
        if dialog.exec_() == QDialog.Accepted:
            self.thrs = dialog.getValue()
            
    def setup_window(self):
        self.setWindowTitle("Smart Sentry")
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setSpacing(10)
        # self.show_threshold_slider()
        self.show_class_selection_dialog()
        self.show_camera_selection_dialog()  # Show camera selection dialog first
        self.create_camera_widgets()  # Create camera labels and workers based on selected cameras

        self.grid_layout = QGridLayout()
        self.grid_layout.setSpacing(10)

        self.main_layout.addLayout(self.grid_layout)
        self.main_layout.addStretch()

        self.bottom_layout = QHBoxLayout()
        self.prev_button = QPushButton("Previous")
        self.prev_button.clicked.connect(self.show_previous_window)
        self.alarm_button = QPushButton("Master Alarm")
        self.alarm_button.setStyleSheet("""QPushButton {background-color: darkgreen;
                                                color: yellow;
                                                font-weight: bold;
                                            }
                                          QPushButton:hover {
                                                background-color: green; /* Darker green for pressed state */
                                            }
                                        """)
        self.alarm_button.clicked.connect(self.turn_global_alarm)
        self.alarm_button.setCheckable(True)
        self.alarm_button.setChecked(False)
        self.next_button = QPushButton("Next")
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.refresh_all)
        self.refresh_btn.setStyleSheet("""
        QPushButton {
            background-color: darkgreen; /* Base background color */
            color: yellow; /* Text color */
            font-weight: bold; /* Bold text */
        }

        QPushButton:hover {
            background-color: yellow; /* Change background color on hover */
            color: darkgreen; /* Change text color on hover */
        }

        QPushButton:pressed {
            background-color: #006400; /* Darker green for pressed state */
            border-color: #004d00; /* Darker border color for pressed state */
        }""")
        self.next_button.clicked.connect(self.show_next_window)
        self.bottom_layout.addWidget(self.prev_button)
        self.bottom_layout.addWidget(self.refresh_btn)
        self.bottom_layout.addWidget(self.alarm_button)
        self.bottom_layout.addWidget(self.next_button)
        self.main_layout.addLayout(self.bottom_layout)
        self.change_view_mode()
        self.welcome_page.ready()

    def refresh_all(self):
        for worker in self.camera_workers:
            if not worker.state_stream:
                worker.start()
                

    def turn_global_alarm(self):
        self.master_btn = not self.master_btn
        if not self.master_btn:
            pygame.mixer.music.rewind()
            pygame.mixer.music.pause()
            self.alarm_button.setStyleSheet("""QPushButton{background-color: darkred;
                                                color: yellow;
                                                font-weight: bold}
                                            QPushButton:hover {
                                                background-color: red; /* Darker green for pressed state */
                                            }
                                            """)
        else:
            self.alarm_button.setStyleSheet("""QPushButton {background-color: darkgreen;
                                                color: yellow;
                                                font-weight: bold}
                                            QPushButton:hover {
                                                background-color: green; /* Darker green for pressed state */
                                            }
                                            """)
        for worker in self.camera_workers:
            if worker.toggle_state != self.master_btn:
                worker.toggleState()
        for label in self.camera_labels:
            if self.master_btn == True:
                label.setStyleSheet('border: 1px solid black;')
            else:
                label.setStyleSheet('border: 3px solid lime;')

    def show_camera_selection_dialog(self):
        dialog = CameraSelectionDialog(self.camera_info, self.welcome_page)
        if dialog.exec_() == QDialog.Accepted:
            selected_cameras = dialog.get_selected_cameras()
            self.camera_info = selected_cameras
            self.num_windows = (len(self.camera_info) + 3) // 4  # Update number of windows based on selected cameras
        dialog.close()

    def show_class_selection_dialog(self):
        dialog = ClassSelectionDialog(self.welcome_page)
        if dialog.exec_() == QDialog.Accepted:
            self.selected_classes = dialog.get_selected_classes()
        dialog.close()

    def file_check_and_create(self, roi_file, entity):
        if not os.path.exists(roi_file):
            with open(roi_file, 'w') as f:
                json.dump({}, f)  # Use f (file object) instead of roi_file
            print(f"File '{roi_file}' created and serialized data has been written to it.")
        else:
            print(f"File '{roi_file}' already exists. Skipping creation.")
        loadedData = None
        try:
            with open(roi_file, 'r') as f:
                loadedData = json.load(f)
        except:
            print(f"{entity}s were found corrupt! Please redefine ROIs at will.")
            with open(roi_file, 'w') as f:
                json.dump({}, f)  # Use f (file object) instead of roi_file
        return loadedData

    def create_camera_widgets(self):
        loadedData = self.file_check_and_create(roi_file, "ROI")
        priority_data = self.file_check_and_create("priority.json", "PRIORITIES")
        for idx, camera_info in enumerate(self.camera_info):
            camera_label = QLabel()
            camera_label.setStyleSheet("border: 1px solid black;")
            self.camera_labels.append(camera_label)
            camera_label.setContextMenuPolicy(Qt.CustomContextMenu)
            camera_label.customContextMenuRequested.connect(lambda point, index=idx: self.showContextMenu(point, index))
            worker = CaptureIpCameraFramesWorker(camera_info['ips'],self.selected_classes,float(camera_info['threshold']))
            if loadedData:
                if camera_info['ips'] in loadedData:
                    with worker.safe_roi:
                        if "roi" in loadedData[camera_info['ips']]:
                            worker.poly = [np.asarray(loadedData[camera_info['ips']]["roi"])]
                        if "ers" in loadedData[camera_info['ips']]:
                            worker.ers = [np.array(x) for x in loadedData[camera_info['ips']]["ers"]]
                        worker.safe_roi.notify_all()
            if priority_data:
                if camera_info["ips"] in priority_data:
                    worker.priority = priority_data[camera_info["ips"]]
            worker.ImageUpdated.connect(lambda image, index=idx: self.update_camera_label(index, image))
            worker.HighlightCamera.connect(lambda highlight, index=idx: self.highlight_camera_label(index, highlight))  # Connect signal to slot
            worker.start()
            self.camera_workers.append(worker)
            camera_label.mouseDoubleClickEvent = lambda event, index=idx: self.show_fullscreen_view(index)
        
        self.alarm_handler = AlarmHandler()
        # Connect alarm signal from worker threads to alarm handler
        for worker in self.camera_workers:
            worker.AlarmTriggered.connect(self.alarm_handler.trigger_alarm)



    def showContextMenu(self, point, index):
        context_menu = QMenu(self)

        # Determine current worker state to set correct action text
        worker = self.camera_workers[index]
        if worker.worker_active:
            action_text = "Turn Off"
        else:
            action_text = "Turn On"
        
        er_text = "Show ERs" if not worker.show_ers else "Hide ERs"
        er_show_hide_action = QAction(er_text, self)

        toggle_action = QAction(action_text, self)
        toggle_action.triggered.connect(lambda: self.toggleWorkerState(index))
        context_menu.addAction(toggle_action)

        define_roi_action = QAction("Define ROI", self)
        define_er_action = QAction("Add ER", self)
        define_remove_er_action = QAction("Remove All ERs", self)

        define_roi_action.triggered.connect(lambda: self.defineROI(index))
        
        def disableDefineROIAction():
            define_roi_action.setEnabled(False)
        
        if worker.screen_mode:
            define_roi_action.setEnabled(False)
        if worker.ers is not None and len(worker.ers) > 10:
            define_er_action.setEnabled(False)

        define_roi_action.triggered.connect(disableDefineROIAction)
        context_menu.addAction(define_roi_action)

        remove_roi_action = QAction("Remove ROI", self)
        if worker.screen_mode:
            remove_roi_action.setEnabled(False)
        
        define_er_action.triggered.connect(lambda: self.addER(index))
        define_remove_er_action.triggered.connect(lambda: self.removeERs(index))
        remove_roi_action.triggered.connect(lambda: self.removeROI(index))
        er_show_hide_action.triggered.connect(lambda: self.show_hide_er(index))
        
        context_menu.addAction(remove_roi_action)

        # Add Priority action
        priority_action = QAction("Set Priority", self)
        context_menu.addAction(priority_action)

        # Create Priority submenu
        t = "  ✔"
        h = "High" + (t if worker.priority == 2 else "")
        m = "Medium" + (t if worker.priority == 1 else "")
        l = "Low" + (t if worker.priority == 0 else "")
        priority_menu = QMenu("Set Priority", self)

        high_action = QAction(h, self)
        medium_action = QAction(m, self)
        low_action = QAction(l, self)

        # Connect the actions to any functions you want to handle them
        high_action.triggered.connect(lambda: self.setPriority(index, 2))
        medium_action.triggered.connect(lambda: self.setPriority(index, 1))
        low_action.triggered.connect(lambda: self.setPriority(index, 0))

        # Add actions to the priority submenu
        priority_menu.addAction(high_action)
        priority_menu.addAction(medium_action)
        priority_menu.addAction(low_action)
        if self.state_admin == 1:
            context_menu.addAction(define_er_action)
            context_menu.addAction(define_remove_er_action)
            context_menu.addAction(er_show_hide_action)
        # Connect the priority action to the submenu
        priority_action.triggered.connect(lambda: priority_menu.exec_(self.camera_labels[index].mapToGlobal(point)))

        context_menu.exec_(self.camera_labels[index].mapToGlobal(point))


    def show_hide_er(self, index):
        self.camera_workers[index].show_ers = not self.camera_workers[index].show_ers

    def setPriority(self, index, value):
        with self.camera_workers[index].safe_p:
            self.camera_workers[index].priority = value
            self.camera_workers[index].safe_p.notify_all()
        p_data = self.file_check_and_create("priority.json", "PRIORITIES")
        p_data[self.camera_workers[index].url] = value
        with open("priority.json", "w") as f:
            json.dump(p_data, f)
        
        


    def addER(self, index):
        if not os.path.exists(roi_file):
            with open(roi_file, 'w') as f:
                json.dump({}, f)  # Use f (file object) instead of roi_file
            print(f"File '{roi_file}' created and serialized data has been written to it.")
        else:
            print(f"File '{roi_file}' already exists. Skipping creation.")
        ex = ImageROIDialog()
        worker = self.camera_workers[index]
        if worker.cap is not None:
            ret, frame = worker.cap.read()
            if ret:
                ex.set_image_from_cv2(frame)
                # Show the application window
                ex.exec_()
                # Get the ROI coordinates after selection completes
                if ex.get_roi_coordinates()[0] is None:
                    return
                with open(roi_file, 'r') as f:
                    loadedData = json.load(f)
                with worker.safe_roi:
                    if worker.ers is None:
                        worker.ers = []
                    worker.ers.append(ex.get_roi_coordinates()[0])
                    if worker.url not in loadedData:
                        loadedData[worker.url] = {}
                    loadedData[worker.url]["ers"] = worker.ers
                    worker.safe_roi.notify_all()
                encodedNumpyData = json.dumps(loadedData, cls=NumpyArrayEncoder)
                with open(roi_file, 'w') as f:
                    f.write(encodedNumpyData)

    def removeERs(self, index):
        worker = self.camera_workers[index]
        if worker.ers is None:
            return
        with worker.safe_roi:
            worker.ers = None
            worker.safe_roi.notify_all()
        with open(roi_file, 'r') as f:
            loadedData = json.load(f)
        for key in loadedData.keys():
            if "ers" not in loadedData[key]:
                continue
            loadedData[key]["ers"] = np.asarray(loadedData[key]["ers"])
            # Process the ROI selection here if needed
        if worker.url in loadedData:
            del loadedData[worker.url]["ers"]
        encodedNumpyData = json.dumps(loadedData, cls=NumpyArrayEncoder)
        with open(roi_file, 'w') as f:
            f.write(encodedNumpyData)

    def removeROI(self, index):
        worker = self.camera_workers[index]
        if worker.poly is None:
            return
        with worker.safe_roi:
            worker.poly = None
            worker.safe_roi.notify_all()
        with open(roi_file, 'r') as f:
            loadedData = json.load(f)
        for key in loadedData.keys():
            if "roi" not in loadedData[key]:
                continue
            loadedData[key]["roi"] = np.asarray(loadedData[key]["roi"])
            # Process the ROI selection here if needed
        if worker.url in loadedData:
            del loadedData[worker.url]["roi"]
        encodedNumpyData = json.dumps(loadedData, cls=NumpyArrayEncoder)
        with open(roi_file, 'w') as f:
            f.write(encodedNumpyData)

    def defineROI(self, index):
        if not os.path.exists(roi_file):
            with open(roi_file, 'w') as f:
                json.dump({}, f)  # Use f (file object) instead of roi_file
            print(f"File '{roi_file}' created and serialized data has been written to it.")
        else:
            print(f"File '{roi_file}' already exists. Skipping creation.")
        ex = ImageROIDialog()
        worker = self.camera_workers[index]
        if worker.cap is not None:
            ret, frame = worker.cap.read()
            if ret:
                ex.set_image_from_cv2(frame)
                # Show the application window
                ex.exec_()
                if (ex.get_roi_coordinates()[0]) is None:
                    return
                # Get the ROI coordinates after selection completes
                with open(roi_file, 'r') as f:
                    loadedData = json.load(f)
                with worker.safe_roi:
                    worker.poly = ex.get_roi_coordinates()
                    print(worker.poly)
                    if worker.url not in loadedData:
                        loadedData[worker.url] = {}
                    loadedData[worker.url]["roi"] = worker.poly[0]
                    worker.safe_roi.notify_all()
                encodedNumpyData = json.dumps(loadedData, cls=NumpyArrayEncoder)
                with open(roi_file, 'w') as f:
                    f.write(encodedNumpyData)


    def toggleWorkerState(self, index):
        worker = self.camera_workers[index]
        worker.toggleState()
        if not worker.worker_active:
            self.camera_labels[index].setStyleSheet('border: 4px solid lime;')
        else:
            self.camera_labels[index].setStyleSheet("border: 1px solid black;")
    def updateWorkerState(self, index, state):
        # Update UI or perform any other actions based on worker state change
        if state:
            print(f"Worker {index} is ON")
        else:
            print(f"Worker {index} is OFF")


    def highlight_camera_label(self, index, highlight):
        if index < len(self.camera_labels):
            camera_label = self.camera_labels[index]
            if highlight:
                camera_label.setStyleSheet('border: 4px solid red;')
            else:
                camera_label.setStyleSheet('border: 1px solid black;')

    def update_camera_label(self, index, frame: QImage):
        frame_t =  frame
        if (self.fn < 10):
            self.fn += 1
            return
        pixmap = QPixmap.fromImage(frame_t)
        if index < len(self.camera_labels):
            camera_label = self.camera_labels[index]
            screen_size = QDesktopWidget().screenGeometry()
            screen_width = screen_size.width()
            screen_height = screen_size.height()

            if self.current_view_mode == '2x2':
                camera_label.setFixedSize((screen_width // 2) - 50, (screen_height // 2) - 70)
            elif self.current_view_mode == '3x3':
                camera_label.setFixedSize((screen_width // 3) - 50, (screen_height // 3) - 55)
            elif self.current_view_mode == '4x4':
                camera_label.setFixedSize((screen_width // 4) - 50, (screen_height // 4) - 40)

            pixmap = pixmap.scaled(camera_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)

            camera_label.setPixmap(pixmap)
            camera_label.setAlignment(Qt.AlignCenter)
            display_ips = self.camera_info[index]['display_ips']
            camera_label.setToolTip(display_ips)

            # Update worker with new frame size requirement
            self.camera_workers[index].update_frame_size(camera_label.size())

    def show_previous_window(self):
        if self.current_window_index > 0:
            self.current_window_index -= 1
            self.show_window(self.current_window_index)

    def show_next_window(self):
        if self.current_window_index < self.num_windows - 1:
            self.current_window_index += 1
            self.show_window(self.current_window_index)

    def show_window(self, window_index):
        self.current_window_index = window_index
        self.num_windows = 0
        if self.current_view_mode == '2x2':
            self.num_windows = (len(self.camera_info) + 3) // 4
        elif self.current_view_mode == '3x3':
            self.num_windows = (len(self.camera_info) + 8) // 9
        else:
            self.num_windows = (len(self.camera_info) + 15) // 16
        for i in reversed(range(self.grid_layout.count())):
            self.grid_layout.itemAt(i).widget().setParent(None)

        if self.current_view_mode == '2x2':
            num_columns = 2
            num_rows = 2
            labels_per_window = 4
        elif self.current_view_mode == '4x4':
            num_columns = 4
            num_rows = 4
            labels_per_window = 16
        elif self.current_view_mode == '3x3':
            num_columns = 3
            num_rows = 3
            labels_per_window = 9
        else:
            num_columns = 4
            num_rows = 4
            labels_per_window = 16

        start_idx = window_index * labels_per_window
        end_idx = min(start_idx + labels_per_window, len(self.camera_labels))

        for idx in range(start_idx, end_idx):
            camera_label = self.camera_labels[idx]
            row = (idx - start_idx) // num_columns
            col = (idx - start_idx) % num_columns
            self.grid_layout.addWidget(camera_label, row, col)

        self.central_widget.adjustSize()

        self.update_buttons_state()

    def change_view_mode(self):
        dialog = ViewDialog()
        dialog.show()
        if dialog.exec_() == QDialog.Accepted:
            self.current_view_mode = dialog.getViewMode()
            self.update_camera_labels_size()
            self.update_camera_workers_size()  # Update worker frame sizes
            self.show_window(self.current_window_index)
        dialog.close()

    def update_camera_labels_size(self):
        for idx in range(len(self.camera_labels)):
            camera_label = self.camera_labels[idx]
            pixmap = camera_label.pixmap()
            if pixmap:
                screen_size = QDesktopWidget().screenGeometry()
                screen_width = screen_size.width()
                screen_height = screen_size.height()

                if self.current_view_mode == '2x2':
                    camera_label.setFixedSize((screen_width // 2) - 50, (screen_height // 2) - 70)
                elif self.current_view_mode == '3x3':
                    camera_label.setFixedSize((screen_width // 3) - 50, (screen_height // 3) - 55)
                elif self.current_view_mode == '4x4':
                    camera_label.setFixedSize((screen_width // 4) - 50, (screen_height // 4) - 40)

                pixmap = pixmap.scaled(camera_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)

                camera_label.setPixmap(pixmap)
                camera_label.setAlignment(Qt.AlignCenter)

    def update_camera_workers_size(self):
        for idx in range(len(self.camera_workers)):
            camera_label = self.camera_labels[idx]
            if camera_label.isVisible():
                self.camera_workers[idx].update_frame_size(camera_label.size())

    def update_buttons_state(self):
        self.prev_button.setEnabled(self.current_window_index > 0)
        self.next_button.setEnabled(self.current_window_index < self.num_windows - 1)

    def show_fullscreen_view(self, index):
        if index < len(self.camera_workers):
            camera_worker = self.camera_workers[index]
            fullscreen_view = FullScreenViewer(camera_worker)
            fullscreen_view.exec_()

class LoginDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.state = 0
        self.setWindowTitle("Login Portal")
        self.setFixedSize(500, 650)
        self.setStyleSheet("background-color: #1e1e1e; color: white; font-size: 28px;")  # Increased font size

        # Center the window on the screen
        screen = QtWidgets.QDesktopWidget().screenGeometry()
        size = self.geometry()
        self.move((screen.width() - size.width()) // 2, (screen.height() - size.height()) // 2)

        # Create layout
        layout = QtWidgets.QVBoxLayout()

        # Logo
        self.logo_label = QtWidgets.QLabel()
        self.logo_pixmap = QtGui.QPixmap("logo4.png")  # Replace with your logo file path
        self.logo_label.setPixmap(self.logo_pixmap.scaled(450, 450, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
        self.logo_label.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(self.logo_label)

        # Title
        title = QtWidgets.QLabel("Smart Sentry")
        title.setAlignment(QtCore.Qt.AlignCenter)
        title.setStyleSheet("font-size: 28px; color: #ffcc00;")  # Increased title font size
        layout.addWidget(title)

        # Password input
        self.password_input = QtWidgets.QLineEdit()
        self.password_input.setEchoMode(QtWidgets.QLineEdit.Password)
        self.password_input.setPlaceholderText("Enter Password (Admin)")
        self.password_input.setStyleSheet("background-color: #333; color: white; padding: 10px; border-radius: 5px; font-size: 28px;")
        layout.addWidget(self.password_input)

        # Login button
        self.login_button = QtWidgets.QPushButton("Login")
        self.login_button.setStyleSheet("""
            QPushButton {
                background-color: #ffcc00; 
                color: black; 
                padding: 10px; 
                border-radius: 5px;
                font-size: 28px;
            }
            QPushButton:hover {
                background-color: #e6b800;  /* Darker yellow on hover */
            }
        """)
        self.login_button.clicked.connect(self.login)
        layout.addWidget(self.login_button)

        # Start as user button
        self.start_user_button = QtWidgets.QPushButton("Continue as User")
        self.start_user_button.setStyleSheet("""
            QPushButton {
                background-color: #444; 
                color: white; 
                padding: 10px; 
                border-radius: 5px; 
                font-size: 28px;
            }
            QPushButton:hover {
                background-color: #666;  /* Lighter gray on hover */
            }
        """)
        self.start_user_button.clicked.connect(self.start_as_user)
        layout.addWidget(self.start_user_button)

        self.setLayout(layout)

        # Install event filter on password input
        self.password_input.installEventFilter(self)

        # Animation for the dialog
        self.animation = QtCore.QPropertyAnimation(self, b"windowOpacity")
        self.animation.setDuration(500)
        self.animation.setStartValue(0)
        self.animation.setEndValue(1)
        self.animation.start()

    def eventFilter(self, source, event):
        if source == self.password_input and event.type() == QtCore.QEvent.KeyPress:
            if event.key() == QtCore.Qt.Key_Return or event.key() == QtCore.Qt.Key_Enter:
                self.login()
                return True  # Stop further processing of the event
        return super().eventFilter(source, event)

    def login(self):
        password = self.password_input.text()
        if password == "paf123":  # Replace with actual password checking logic
            self.state = 1
            self.accept()  # Close the dialog with accepted state
        else:
            QtWidgets.QMessageBox.warning(self, "Error", "Incorrect Password.")

    def start_as_user(self):
        self.state = 2
        self.accept()  # Close the dialog with accepted state

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5()) 
    print(torch.cuda.is_available())
    window = None
    dialog = LoginDialog()
    if dialog.exec_() == QtWidgets.QDialog.Accepted:
        if dialog.state == 1:
            window = MainWindow(1)
        elif dialog.state == 2:
            window = MainWindow(0)
        window.show()
        app.exec_()
