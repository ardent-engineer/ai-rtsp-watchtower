from PyQt5.QtWidgets import QLabel, QSlider
from PyQt5.QtGui import QPainter, QPen
from PyQt5.QtCore import pyqtSignal, Qt, QTimer

class FloatSlider(QSlider):
    valueChangedFloat = pyqtSignal(float)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setOrientation(Qt.Horizontal)
        self.setRange(0, 100)  # Set range from 0 to 100 (corresponding to 0.0 to 1.0)
        self.setSingleStep(1)  # Set the step size
        self.setValue(45)  # Initial value at 0.45
        self.valueChanged.connect(self.emitFloatValue)

    def emitFloatValue(self, value):
        float_value = value / 100.0  # Convert integer value to float between 0.0 and 1.0
        self.valueChangedFloat.emit(float_value)

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
        if self.parent() and hasattr(self.parent(), 'show_roi_coordinates'):
            self.parent().show_roi_coordinates()
        self.accept_timer.stop()