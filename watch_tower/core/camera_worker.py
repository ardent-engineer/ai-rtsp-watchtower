import threading
import numpy as np
import torch
import cv2
from mss import mss
from PIL import Image

from PyQt5.QtCore import QThread, pyqtSignal, QSize
from PyQt5.QtGui import QImage

# Relative import to get the AsyncRead class from the utils module
from ..utils.async_read import AsyncRead


class CaptureIpCameraFramesWorker(QThread):
    ImageUpdated = pyqtSignal(QImage)
    HighlightCamera = pyqtSignal(bool)
    toggleStateChanged = pyqtSignal(bool)
    AlarmTriggered = pyqtSignal(int)

    def __init__(self, url, selected_classes, threshold, config, parent=None):
        super().__init__(parent)
        self.url = url
        self.config = config
        self.__thread_active = True
        self.__thread_pause = False
        self.cap = None
        self.priority = 0
        self.screen = None
        self.screen_mode = self.urls_screen()
        self.frame_size = QSize(640, 480)
        self.selected_classes = {}
        
        # --- CORRECTED MODEL LOADING ---
        # Each worker loads its own instance of the model for thread safety.
        g_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = torch.hub.load(
            self.config['paths']['yolo_dir'], 
            'custom',
            source='local', 
            path=self.config['paths']['model'],
            device=g_device, 
            force_reload=True
        )
        self.model.conf = threshold
        self.model.eval()
        if g_device == "cuda":
            self.model.cuda()

        self.classes = {}
        self.time = []
        self.poly = None
        self.ers = None
        self.worker_active = True
        self.toggle_state = True
        self.frames_per_detection = 3
        self.detect_now = 0
        
        # Load the 'no_signal' image once during initialization
        no_signal_path = self.config['paths']['assets']['images']['no_signal']
        img_array = np.array(Image.open(no_signal_path))
        self.no_signal_img = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

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
        print(f"[{self.url}] attempting restart!")
        if self.screen_mode:
            try:
                with mss() as sct:
                    monitor = sct.monitors[self.screen]
                    while self.__thread_active:
                        screenshot = sct.grab(monitor)
                        frame = np.array(screenshot, dtype=np.uint8)
                        cv_rgb_image = self.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        qt_image = QImage(cv_rgb_image.data, cv_rgb_image.shape[1], cv_rgb_image.shape[0], cv_rgb_image.shape[1]*3, QImage.Format_RGB888)
                        self.ImageUpdated.emit(qt_image)
            except Exception as e:
                print(f"Monitor: {self.screen} is not available! Error: {e}")
                img = self.no_signal_img
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
                img = self.no_signal_img
                h, w, c = img.shape
                bytes_per = w * c
                qt_rgb_ = QImage(img.data, w, h, bytes_per, QImage.Format_RGB888)
                self.ImageUpdated.emit(qt_rgb_)
                self.cap.release()
                self.stop()
                return
            
            self.cap = AsyncRead(self.cap)
            self.state_stream = True
            while self.__thread_active:
                if not self.__thread_pause:
                    
                    ret, frame = self.cap.read(wait=True, timeout=1.0) # Use a 1-second timeout

                    if self.__thread_active is False: # Add an immediate check after the wait
                        break

                    if frame is not None and ret:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        cv_rgb_image = self.detect(frame)
                        qt_image = QImage(cv_rgb_image.data, cv_rgb_image.shape[1], cv_rgb_image.shape[0], cv_rgb_image.shape[1]*3, QImage.Format_RGB888)
                        self.ImageUpdated.emit(qt_image)
                    else:
                        if self.state_stream is False:
                            break
                        time.sleep(0.1) 
            
            self.state_stream = False
            print("stream exit occurred!")

    def any_one_rectangle_corners_inside_polygon(self, x1, y1, x2, y2, polygon_corners):
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
        if len(polygon_corners) == 0:
            return True
        x, y = point
        polygon_corners = np.asarray(polygon_corners) # Ensure it's a numpy array
        x_coords, y_coords = polygon_corners[:, 0], polygon_corners[:, 1]

        if np.min(x_coords) <= x <= np.max(x_coords) and np.min(y_coords) <= y <= np.max(y_coords):
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
        self.__thread_active = False # Signal thread to stop
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
        self.toggleStateChanged.emit(self.toggle_state)
        self.worker_active = not self.worker_active

    def isToggleOn(self):
        return self.toggle_state