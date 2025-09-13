import sys
import csv
import json
import os
import pygame
import numpy as np

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QLabel, QGridLayout, QPushButton,
    QHBoxLayout, QVBoxLayout, QDesktopWidget, QMessageBox, QMenu, QAction
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

# --- Local Project Imports ---
from .core.camera_worker import CaptureIpCameraFramesWorker
from .core.alarm_handler import AlarmHandler
from .ui.dialogs import (
    WelcomePage, CameraSelectionDialog, ClassSelectionDialog, 
    ViewDialog, ImageROIDialog, FullScreenViewer
)
from .utils.json_encoder import NumpyArrayEncoder


class MainWindow(QMainWindow):
    def __init__(self, state, config):
        super(MainWindow, self).__init__()
        self.state_admin = state
        self.config = config  # <-- Store the config
        
        self.fn = 0
        self.camera_info = []
        self.camera_workers = []
        self.camera_labels = []
        self.current_window_index = 0
        self.num_windows = 0
        self.current_view_mode = '4x4'
        self.master_btn = True
        
        screen_size = QDesktopWidget().screenGeometry()
        self.resize(screen_size.width(), screen_size.height())
        
        # The WelcomePage is shown from __main__.py before this window is created
        self.load_camera_info_from_csv()
        self.setup_window()
        self.center()
        self.show_window(0) # Show the first page of cameras initially

    def start_application(self):
        self.welcome_page.close()
        self.show()

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def load_camera_info_from_csv(self):
        try:
            with open(self.config['paths']['sources_csv'], newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    if not 0.0 <= float(row['threshold']) <= 1.0:
                        row['threshold'] = 0.5
                    self.camera_info.append({
                        'threshold': row['threshold'],
                        'display_ips': row['display_ips'],
                        'ips': row['ips']
                    })
        except FileNotFoundError:
            QMessageBox.critical(self, "File Error", f"{self.config['paths']['sources_csv']} not found.")
            sys.exit(1)

    def setup_window(self):
        self.setWindowTitle("Smart Sentry")
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        self.welcome_page = WelcomePage(self.config, self) # For parenting dialogs
        self.welcome_page.hide() # We only need it as a parent

        self.show_class_selection_dialog()
        self.show_camera_selection_dialog()
        self.create_camera_widgets()

        self.grid_layout = QGridLayout()
        self.grid_layout.setSpacing(10)
        self.main_layout.addLayout(self.grid_layout)
        self.main_layout.addStretch()

        self.bottom_layout = QHBoxLayout()
        self.prev_button = QPushButton("Previous")
        self.prev_button.clicked.connect(self.show_previous_window)
        self.alarm_button = QPushButton("Master Alarm")
        self.alarm_button.setStyleSheet("QPushButton {background-color: darkgreen; color: yellow; font-weight: bold;} QPushButton:hover {background-color: green;}")
        self.alarm_button.clicked.connect(self.turn_global_alarm)
        self.next_button = QPushButton("Next")
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.refresh_all)
        self.refresh_btn.setStyleSheet("QPushButton {background-color: darkgreen; color: yellow; font-weight: bold;} QPushButton:hover {background-color: yellow; color: darkgreen;}")
        self.next_button.clicked.connect(self.show_next_window)
        
        self.bottom_layout.addWidget(self.prev_button)
        self.bottom_layout.addWidget(self.refresh_btn)
        self.bottom_layout.addWidget(self.alarm_button)
        self.bottom_layout.addWidget(self.next_button)
        self.main_layout.addLayout(self.bottom_layout)
        
        self.change_view_mode()

    def refresh_all(self):
        for worker in self.camera_workers:
            if not worker.state_stream and not worker.isRunning():
                worker.start()

    def turn_global_alarm(self):
        self.master_btn = not self.master_btn
        if not self.master_btn:
            pygame.mixer.stop() # Stops all sounds on all channels
            self.alarm_button.setStyleSheet("QPushButton{background-color: darkred; color: yellow; font-weight: bold} QPushButton:hover {background-color: red;}")
        else:
            self.alarm_button.setStyleSheet("QPushButton {background-color: darkgreen; color: yellow; font-weight: bold} QPushButton:hover {background-color: green;}")
        
        for worker in self.camera_workers:
            if worker.isToggleOn() != self.master_btn:
                worker.toggleState()
        
        for label in self.camera_labels:
            stylesheet = 'border: 3px solid lime;' if not self.master_btn else 'border: 1px solid black;'
            label.setStyleSheet(stylesheet)

    def show_camera_selection_dialog(self):
        dialog = CameraSelectionDialog(self.camera_info, self.welcome_page)
        if dialog.exec_() == dialog.Accepted:
            self.camera_info = dialog.get_selected_cameras()

    def show_class_selection_dialog(self):
        dialog = ClassSelectionDialog(self.welcome_page)
        if dialog.exec_() == dialog.Accepted:
            self.selected_classes = dialog.get_selected_classes()

    def file_check_and_create(self, file_path, entity_name):
        if not os.path.exists(file_path):
            with open(file_path, 'w') as f: json.dump({}, f)
        try:
            with open(file_path, 'r') as f: return json.load(f)
        except json.JSONDecodeError:
            print(f"{entity_name} file was found corrupt! Creating a new one.")
            with open(file_path, 'w') as f: json.dump({}, f)
            return {}

    def create_camera_widgets(self):
        roi_data = self.file_check_and_create(self.config['paths']['roi_file'], "ROI")
        priority_data = self.file_check_and_create(self.config['paths']['priority_file'], "PRIORITIES")

        for idx, cam_info in enumerate(self.camera_info):
            label = QLabel()
            label.setStyleSheet("border: 1px solid black;")
            label.setContextMenuPolicy(Qt.CustomContextMenu)
            label.customContextMenuRequested.connect(lambda point, index=idx: self.showContextMenu(point, index))
            self.camera_labels.append(label)

            worker = CaptureIpCameraFramesWorker(
                url=cam_info['ips'],
                selected_classes=self.selected_classes,
                threshold=float(cam_info['threshold']),
                config=self.config
            )

            if roi_data and cam_info['ips'] in roi_data:
                with worker.safe_roi:
                    if "roi" in roi_data[cam_info['ips']]: worker.poly = [np.asarray(roi_data[cam_info['ips']]["roi"])]
                    if "ers" in roi_data[cam_info['ips']]: worker.ers = [np.array(x) for x in roi_data[cam_info['ips']]["ers"]]
                    worker.safe_roi.notify_all()
            
            if priority_data and cam_info["ips"] in priority_data:
                worker.priority = priority_data[cam_info["ips"]]

            worker.ImageUpdated.connect(lambda image, index=idx: self.update_camera_label(index, image))
            worker.HighlightCamera.connect(lambda highlight, index=idx: self.highlight_camera_label(index, highlight))
            worker.start()
            self.camera_workers.append(worker)
            label.mouseDoubleClickEvent = lambda event, index=idx: self.show_fullscreen_view(index)
        
        sound_paths = [self.config['paths']['assets']['sounds'][f'alarm_{i}'] for i in range(3)]
        self.alarm_handler = AlarmHandler(
            sound_paths=sound_paths,
            cooldown_ms=self.config['alarm_settings']['alarm_cooldown_ms'],
            grey_zone_ms=self.config['alarm_settings']['grey_zone_ms'],
            black_zone_ms=self.config['alarm_settings']['black_zone_ms']
        )
        for worker in self.camera_workers:
            worker.AlarmTriggered.connect(self.alarm_handler.trigger_alarm)

    def showContextMenu(self, point, index):
        context_menu = QMenu(self)
        worker = self.camera_workers[index]

        action_text = "Turn Off" if worker.worker_active else "Turn On"
        toggle_action = QAction(action_text, self)
        toggle_action.triggered.connect(lambda: self.toggleWorkerState(index))
        context_menu.addAction(toggle_action)

        define_roi_action = QAction("Define ROI", self)
        define_roi_action.triggered.connect(lambda: self.defineROI(index))
        if worker.screen_mode: define_roi_action.setEnabled(False)
        context_menu.addAction(define_roi_action)

        remove_roi_action = QAction("Remove ROI", self)
        remove_roi_action.triggered.connect(lambda: self.removeROI(index))
        if worker.screen_mode: remove_roi_action.setEnabled(False)
        context_menu.addAction(remove_roi_action)

        # Priority Submenu
        priority_menu = QMenu("Set Priority", self)
        priorities = {"High": 2, "Medium": 1, "Low": 0}
        for name, level in priorities.items():
            text = f"{name}{'  ✔' if worker.priority == level else ''}"
            action = QAction(text, self)
            action.triggered.connect(lambda checked, i=index, l=level: self.setPriority(i, l))
            priority_menu.addAction(action)
        context_menu.addMenu(priority_menu)

        if self.state_admin == 1:
            context_menu.addSeparator()
            er_text = "Show ERs" if not worker.show_ers else "Hide ERs"
            er_show_hide_action = QAction(er_text, self)
            er_show_hide_action.triggered.connect(lambda: self.show_hide_er(index))
            context_menu.addAction(er_show_hide_action)

            define_er_action = QAction("Add ER", self)
            if worker.ers is not None and len(worker.ers) > 10: define_er_action.setEnabled(False)
            define_er_action.triggered.connect(lambda: self.addER(index))
            context_menu.addAction(define_er_action)

            define_remove_er_action = QAction("Remove All ERs", self)
            define_remove_er_action.triggered.connect(lambda: self.removeERs(index))
            context_menu.addAction(define_remove_er_action)
            
        context_menu.exec_(self.camera_labels[index].mapToGlobal(point))

    def show_hide_er(self, index):
        self.camera_workers[index].show_ers = not self.camera_workers[index].show_ers

    def setPriority(self, index, value):
        worker = self.camera_workers[index]
        with worker.safe_p:
            worker.priority = value
            worker.safe_p.notify_all()
        
        p_data = self.file_check_and_create(self.config['paths']['priority_file'], "PRIORITIES")
        p_data[worker.url] = value
        with open(self.config['paths']['priority_file'], "w") as f:
            json.dump(p_data, f)
    
    def _update_roi_file(self, url, key, value):
        roi_file_path = self.config['paths']['roi_file']
        data = self.file_check_and_create(roi_file_path, "ROI")
        if url not in data: data[url] = {}
        if value is None:
            if key in data[url]: del data[url][key]
        else:
            data[url][key] = value
        
        with open(roi_file_path, 'w') as f:
            json.dump(data, f, cls=NumpyArrayEncoder)

    def addER(self, index):
        worker = self.camera_workers[index]
        if worker.cap is None: return
        ret, frame = worker.cap.read()
        if not ret: return

        ex = ImageROIDialog()
        ex.set_image_from_cv2(frame)
        if ex.exec_() == ex.Accepted and ex.get_roi_coordinates()[0] is not None:
            with worker.safe_roi:
                if worker.ers is None: worker.ers = []
                worker.ers.append(ex.get_roi_coordinates()[0])
                self._update_roi_file(worker.url, "ers", worker.ers)
                worker.safe_roi.notify_all()
    
    def removeERs(self, index):
        worker = self.camera_workers[index]
        with worker.safe_roi:
            worker.ers = None
            self._update_roi_file(worker.url, "ers", None)
            worker.safe_roi.notify_all()

    def removeROI(self, index):
        worker = self.camera_workers[index]
        with worker.safe_roi:
            worker.poly = None
            self._update_roi_file(worker.url, "roi", None)
            worker.safe_roi.notify_all()

    def defineROI(self, index):
        worker = self.camera_workers[index]
        if worker.cap is None: return
        ret, frame = worker.cap.read()
        if not ret: return

        ex = ImageROIDialog()
        ex.set_image_from_cv2(frame)
        if ex.exec_() == ex.Accepted and ex.get_roi_coordinates()[0] is not None:
            with worker.safe_roi:
                worker.poly = ex.get_roi_coordinates()
                self._update_roi_file(worker.url, "roi", worker.poly[0])
                worker.safe_roi.notify_all()

    def toggleWorkerState(self, index):
        worker = self.camera_workers[index]
        worker.toggleState()
        stylesheet = 'border: 4px solid lime;' if not worker.worker_active else "border: 1px solid black;"
        self.camera_labels[index].setStyleSheet(stylesheet)
    
    def highlight_camera_label(self, index, highlight):
        if index < len(self.camera_labels):
            label = self.camera_labels[index]
            current_style = label.styleSheet()
            if 'lime' in current_style: return # Don't override manual off state
            stylesheet = 'border: 4px solid red;' if highlight else 'border: 1px solid black;'
            label.setStyleSheet(stylesheet)

    def update_camera_label(self, index, frame: QImage):
        if self.fn < 10: # Initial frame buffer
            self.fn += 1
            return
        pixmap = QPixmap.fromImage(frame)
        if index < len(self.camera_labels):
            label = self.camera_labels[index]
            pixmap = pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            label.setPixmap(pixmap)
            label.setAlignment(Qt.AlignCenter)
            label.setToolTip(self.camera_info[index]['display_ips'])

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
        view_map = {'2x2': 4, '3x3': 9, '4x4': 16}
        labels_per_window = view_map.get(self.current_view_mode, 16)
        
        self.num_windows = (len(self.camera_info) + labels_per_window - 1) // labels_per_window

        # Clear grid layout
        for i in reversed(range(self.grid_layout.count())): 
            self.grid_layout.itemAt(i).widget().setParent(None)

        num_columns = int(labels_per_window**0.5)
        start_idx = window_index * labels_per_window
        end_idx = min(start_idx + labels_per_window, len(self.camera_labels))

        for idx in range(start_idx, end_idx):
            row = (idx - start_idx) // num_columns
            col = (idx - start_idx) % num_columns
            self.grid_layout.addWidget(self.camera_labels[idx], row, col)

        self.update_buttons_state()

    def change_view_mode(self):
        dialog = ViewDialog()
        if dialog.exec_() == dialog.Accepted:
            self.current_view_mode = dialog.getViewMode()
            self.update_camera_labels_size()
            self.show_window(self.current_window_index)

    def update_camera_labels_size(self):
        screen_size = QDesktopWidget().screenGeometry()
        width, height = screen_size.width(), screen_size.height()
        size_map = {
            '2x2': ((width // 2) - 50, (height // 2) - 70),
            '3x3': ((width // 3) - 50, (height // 3) - 55),
            '4x4': ((width // 4) - 50, (height // 4) - 40)
        }
        w, h = size_map.get(self.current_view_mode, size_map['4x4'])
        for label in self.camera_labels:
            label.setFixedSize(w, h)

    def update_buttons_state(self):
        self.prev_button.setEnabled(self.current_window_index > 0)
        self.next_button.setEnabled(self.current_window_index < self.num_windows - 1)

    def show_fullscreen_view(self, index):
        if index < len(self.camera_workers):
            fullscreen_view = FullScreenViewer(self.camera_workers[index])
            fullscreen_view.exec_()
            
    def closeEvent(self, event):
        for worker in self.camera_workers:
            worker.stop()
        super().closeEvent(event)