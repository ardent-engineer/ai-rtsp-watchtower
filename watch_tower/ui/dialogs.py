import cv2
import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton,
    QHBoxLayout, QVBoxLayout, QDialog, QRadioButton, QDialogButtonBox,
    QCheckBox, QScrollArea, QDesktopWidget, QLineEdit
)
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import Qt, QSize

# Relative imports from other modules in our application package
from .widgets import FloatSlider, ImageLabel


class WelcomePage(QDialog):
    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = config
        self.setWindowTitle("Welcome to Smart Sentry")
        self.setStyleSheet(f"""
            QDialog {{
                border-image: url('{self.config['paths']['assets']['images']['welcome_bg']}') 0 0 0 0 stretch stretch;
            }}
        """)
        
        self.logo_label = QLabel()
        pixmap = QPixmap(self.config['paths']['assets']['images']['logo'])
        self.logo_label.setPixmap(pixmap)
        self.logo_label.setAlignment(Qt.AlignCenter)
        self.logo_label.setStyleSheet("background-color: rgba(0, 0, 0, 0);")
        
        self.start_button = QPushButton("Launching System! Please Wait")
        self.start_button.setStyleSheet("""
            QPushButton {
                background-color: #D32F2F; border: none; color: white; padding: 15px 32px;
                text-align: center; font-size: 28px; margin: 4px 2px; border-radius: 8px;
            }
            QPushButton:hover { background-color: #C62828; border: 2px solid yellow; }
        """)
        self.start_button.setEnabled(False)
        self.start_button.clicked.connect(self.accept)

        main_layout = QVBoxLayout()
        main_layout.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.logo_label)
        main_layout.addWidget(self.start_button)
        self.setLayout(main_layout)
        self.setGeometry(QApplication.desktop().availableGeometry())

    def ready(self):
        self.start_button.setText("Press to Start!")
        self.start_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50; border: none; color: white; padding: 15px 32px;
                text-align: center; font-size: 28px; margin: 4px 2px; border-radius: 8px;
            }
            QPushButton:hover { background-color: #45a049; border: 2px solid yellow; }
        """)
        self.start_button.setEnabled(True)
        QtWidgets.QApplication.processEvents()

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

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
        checkbox_font.setPointSize(12)

        for info in self.camera_info:
            checkbox = QCheckBox(info)
            checkbox.setFont(checkbox_font)
            checkbox.setStyleSheet("""
                QCheckBox { padding: 10px; border: 2px solid black; border-radius: 5px; }
                QCheckBox::indicator { width: 20px; height: 20px; border-radius: 5px; }
                QCheckBox::indicator:checked { background-color: green; border: 2px solid darkgreen; }
                QCheckBox::indicator:unchecked { background-color: white; border: 2px solid black; }
                QCheckBox::indicator:checked:hover { background-color: lightgreen; }
            """)
            checkbox.setIconSize(QSize(0, 0))
            checkbox.setChecked(True)
            scroll_layout.addWidget(checkbox)
            self.camera_checkboxes.append(checkbox)

        scroll_widget.setLayout(scroll_layout)
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        main_layout = QVBoxLayout()
        main_layout.addWidget(scroll_area)
        main_layout.addWidget(button_box)
        self.setLayout(main_layout)
        self.resize(700, 400)
        self.center()

    def get_selected_classes(self):
        selected_classes = set()
        if self.camera_checkboxes[0].isChecked(): selected_classes.add("Animal")
        if self.camera_checkboxes[1].isChecked(): selected_classes.add("Person")
        if self.camera_checkboxes[2].isChecked(): selected_classes.add("Vehicle")
        return selected_classes

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

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
        checkbox_font.setPointSize(12)

        for info in self.camera_info:
            checkbox = QCheckBox(info['display_ips'])
            checkbox.setFont(checkbox_font)
            checkbox.setStyleSheet("""
                QCheckBox { padding: 10px; border: 2px solid black; border-radius: 5px; }
                QCheckBox::indicator { width: 20px; height: 20px; border-radius: 5px; }
                QCheckBox::indicator:checked { background-color: green; border: 2px solid darkgreen; }
                QCheckBox::indicator:unchecked { background-color: white; border: 2px solid black; }
                QCheckBox::indicator:checked:hover { background-color: lightgreen; }
            """)
            checkbox.setIconSize(QSize(0, 0))
            checkbox.setChecked(True)
            scroll_layout.addWidget(checkbox)
            self.camera_checkboxes.append(checkbox)

        scroll_widget.setLayout(scroll_layout)
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        main_layout = QVBoxLayout()
        main_layout.addWidget(scroll_area)
        main_layout.addWidget(button_box)
        self.setLayout(main_layout)
        self.resize(700, 500)
        self.center()

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def get_selected_cameras(self):
        selected_cameras = [self.camera_info[idx] for idx, cb in enumerate(self.camera_checkboxes) if cb.isChecked()]
        return selected_cameras

class ViewDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Choose View Mode")
        self.radio_2x2 = QRadioButton("2x2 View")
        self.radio_3x3 = QRadioButton("3x3 View")
        self.radio_4x4 = QRadioButton("4x4 View")
        self.radio_4x4.setChecked(True)
        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout = QVBoxLayout()
        layout.addWidget(self.radio_2x2)
        layout.addWidget(self.radio_3x3)
        layout.addWidget(self.radio_4x4)
        layout.addWidget(self.button_box)
        self.setLayout(layout)
        self.setStyleSheet("""
            QRadioButton { padding: 10px; border: 2px solid black; border-radius: 5px; }
            QRadioButton::indicator { width: 20px; height: 20px; border-radius: 5px; }
            QRadioButton::indicator:checked { background-color: green; border: 2px solid darkgreen; }
            QRadioButton::indicator:unchecked { background-color: white; border: 2px solid black; }
            QRadioButton::indicator:checked:hover { background-color: lightgreen; }
        """)
        radio_font = QFont()
        radio_font.setPointSize(12)
        self.radio_2x2.setFont(radio_font)
        self.radio_3x3.setFont(radio_font)
        self.radio_4x4.setFont(radio_font)
        self.resize(500, 300)
        self.center()

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def getViewMode(self):
        if self.radio_2x2.isChecked(): return '2x2'
        elif self.radio_3x3.isChecked(): return '3x3'
        elif self.radio_4x4.isChecked(): return '4x4'
        else: return None

class FullScreenViewer(QDialog):
    def __init__(self, camera_worker):
        super().__init__()
        self.setWindowTitle("Fullscreen View")
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint)
        self.label = QLabel()
        self.label.setAlignment(Qt.AlignCenter)
        self.camera_worker = camera_worker
        self.camera_worker.ImageUpdated.connect(self.update_image)
        if not self.camera_worker.isRunning():
            self.camera_worker.start()
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        self.setLayout(layout)
        screen_size = QDesktopWidget().screenGeometry()
        self.resize(screen_size.width(), screen_size.height())

    def update_image(self, frame: QImage):
        pixmap = QPixmap.fromImage(frame)
        scaled_pixmap = pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.label.setPixmap(scaled_pixmap)

    def closeEvent(self, event):
        self.camera_worker.ImageUpdated.disconnect(self.update_image)
        # We don't stop the worker, just disconnect from its signal
        super().closeEvent(event)

class FloatSliderDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Detection Threshold")
        layout = QVBoxLayout(self)
        self.slider = FloatSlider()
        self.slider.valueChangedFloat.connect(self.onSliderValueChangedFloat)
        layout.addWidget(self.slider)
        self.label = QLabel(f"Value: {self.slider.value() / 100.0:.2f}")
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
        return self.slider.value() / 100.0

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

class ImageROIDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image_label = ImageLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        self.setLayout(layout)
        self.setWindowTitle('Polygon ROI Selection')
        self.roi_coordinates = None
        self.original_frame_size = None

    def set_image_from_cv2(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channel = frame.shape
        self.original_frame_size = (width, height)
        max_width = QApplication.desktop().screenGeometry().width() * 0.5
        max_height = QApplication.desktop().screenGeometry().height() * 0.5
        scale_factor = min(max_width / width, max_height / height)
        resized_width = int(width * scale_factor)
        resized_height = int(height * scale_factor)
        qimage = QImage(frame_rgb.data, width, height, width * channel, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qimage).scaled(resized_width, resized_height, Qt.KeepAspectRatio))
        self.resize(resized_width, resized_height)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            self.show_roi_coordinates()
        else:
            super().keyPressEvent(event)

    def show_roi_coordinates(self):
        if len(self.image_label.points) >= 3:
            roi_coordinates = np.array([[p.x() * self.original_frame_size[0] // self.image_label.width(),
                                         p.y() * self.original_frame_size[1] // self.image_label.height()]
                                        for p in self.image_label.points])
            self.roi_coordinates = roi_coordinates.astype(int)
            print(f"ROI Coordinates: {self.roi_coordinates}")
            self.accept()

    def get_roi_coordinates(self):
        return [self.roi_coordinates]

class LoginDialog(QtWidgets.QDialog):
    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = config
        self.state = 0
        self.setWindowTitle("Login Portal")
        self.setFixedSize(500, 650)
        self.setStyleSheet("background-color: #1e1e1e; color: white; font-size: 28px;")
        screen = QtWidgets.QDesktopWidget().screenGeometry()
        size = self.geometry()
        self.move((screen.width() - size.width()) // 2, (screen.height() - size.height()) // 2)

        layout = QtWidgets.QVBoxLayout()
        self.logo_label = QtWidgets.QLabel()
        self.logo_pixmap = QtGui.QPixmap(self.config['paths']['assets']['images']['logo'])
        self.logo_label.setPixmap(self.logo_pixmap.scaled(450, 450, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
        self.logo_label.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(self.logo_label)

        title = QtWidgets.QLabel("Smart Sentry")
        title.setAlignment(QtCore.Qt.AlignCenter)
        title.setStyleSheet("font-size: 28px; color: #ffcc00;")
        layout.addWidget(title)

        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.Password)
        self.password_input.setPlaceholderText("Enter Password (Admin)")
        self.password_input.setStyleSheet("background-color: #333; color: white; padding: 10px; border-radius: 5px; font-size: 28px;")
        layout.addWidget(self.password_input)

        self.login_button = QPushButton("Login")
        self.login_button.setStyleSheet("""
            QPushButton { background-color: #ffcc00; color: black; padding: 10px; border-radius: 5px; font-size: 28px; }
            QPushButton:hover { background-color: #e6b800; }
        """)
        self.login_button.clicked.connect(self.login)
        layout.addWidget(self.login_button)

        self.start_user_button = QPushButton("Continue as User")
        self.start_user_button.setStyleSheet("""
            QPushButton { background-color: #444; color: white; padding: 10px; border-radius: 5px; font-size: 28px; }
            QPushButton:hover { background-color: #666; }
        """)
        self.start_user_button.clicked.connect(self.start_as_user)
        layout.addWidget(self.start_user_button)
        self.setLayout(layout)
        self.password_input.installEventFilter(self)
        self.animation = QtCore.QPropertyAnimation(self, b"windowOpacity")
        self.animation.setDuration(500)
        self.animation.setStartValue(0)
        self.animation.setEndValue(1)
        self.animation.start()

    def eventFilter(self, source, event):
        if source == self.password_input and event.type() == QtCore.QEvent.KeyPress:
            if event.key() == QtCore.Qt.Key_Return or event.key() == QtCore.Qt.Key_Enter:
                self.login()
                return True
        return super().eventFilter(source, event)

    def login(self):
        if self.password_input.text() == "123":
            self.state = 1
            self.accept()
        else:
            QtWidgets.QMessageBox.warning(self, "Error", "Incorrect Password.")

    def start_as_user(self):
        self.state = 2
        self.accept()