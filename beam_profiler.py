#!/usr/bin/env python3
import sys
import cv2


import numpy as np
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QDoubleSpinBox,
    QSpinBox,
    QCheckBox,
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt

# Import your custom modules (make sure PYTHONPATH is set appropriately)
from blob_detector import blob_detector
from basler import Baslera2A4504


class BeamProfilerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Beam Profiler GUI")

        # Initialize camera and set a default ROI.
        self.camera = Baslera2A4504()
        self.roi_x = 0
        self.roi_y = 0
        self.roi_w = 2000
        self.roi_h = 2000
        self.camera.CROI((self.roi_x, self.roi_y, self.roi_w, self.roi_h))

        self.fixed_blobs = []  # list to hold fixed blob dictionaries
        self.current_spots = []  # updated in update_frame() with current detected spots
        self.video_label.mousePressEvent = self.handle_mouse_press

        # Create the main widget and layouts.
        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)
        main_layout = QHBoxLayout()
        self.main_widget.setLayout(main_layout)

        # Video display area.
        self.video_label = QLabel()
        self.video_label.setFixedSize(800, 800)  # adjust as needed
        main_layout.addWidget(self.video_label)

        # Controls panel.
        control_layout = QVBoxLayout()
        main_layout.addLayout(control_layout)

        # Auto exposure checkbox.
        self.auto_exposure_checkbox = QCheckBox("Auto Exposure")
        self.auto_exposure_checkbox.setChecked(False)
        control_layout.addWidget(self.auto_exposure_checkbox)

        # Exposure control (manual mode).
        self._least_exposure = 20
        self._max_exposure = 1e4
        self.exposure_spin = QDoubleSpinBox()
        self.exposure_spin.setPrefix("Exposure: ")
        self.exposure_spin.setRange(self._least_exposure, self._max_exposure)
        self.exposure_spin.setValue(200)  # default value
        control_layout.addWidget(self.exposure_spin)

        # Gain control (manual mode).
        self._least_gain = 0
        self._max_gain = 10
        self.gain_spin = QDoubleSpinBox()
        self.gain_spin.setPrefix("Gain: ")
        self.gain_spin.setRange(self._least_gain, self._max_gain)
        self.gain_spin.setValue(0)  # default value
        control_layout.addWidget(self.gain_spin)

        # Blob detection threshold controls.
        # In BeamProfilerGUI.__init__ after other controls:
        self.blob_min_threshold_spin = QSpinBox()
        self.blob_min_threshold_spin.setPrefix("Blob Min Thresh: ")
        self.blob_min_threshold_spin.setRange(0, 255)
        self.blob_min_threshold_spin.setValue(10)
        control_layout.addWidget(self.blob_min_threshold_spin)

        self.blob_max_threshold_spin = QSpinBox()
        self.blob_max_threshold_spin.setPrefix("Blob Max Thresh: ")
        self.blob_max_threshold_spin.setRange(0, 255)
        self.blob_max_threshold_spin.setValue(200)
        control_layout.addWidget(self.blob_max_threshold_spin)

        # ROI controls.
        roi_layout = QGridLayout()
        control_layout.addLayout(roi_layout)
        roi_layout.addWidget(QLabel("ROI X:"), 0, 0)
        self.roi_x_spin = QSpinBox()
        self.roi_x_spin.setRange(0, 5000)
        self.roi_x_spin.setValue(self.roi_x)
        roi_layout.addWidget(self.roi_x_spin, 0, 1)

        roi_layout.addWidget(QLabel("ROI Y:"), 1, 0)
        self.roi_y_spin = QSpinBox()
        self.roi_y_spin.setRange(0, 5000)
        self.roi_y_spin.setValue(self.roi_y)
        roi_layout.addWidget(self.roi_y_spin, 1, 1)

        roi_layout.addWidget(QLabel("ROI Width:"), 2, 0)
        self.roi_w_spin = QSpinBox()
        self.roi_w_spin.setRange(1, 5000)
        self.roi_w_spin.setValue(self.roi_w)
        roi_layout.addWidget(self.roi_w_spin, 2, 1)

        roi_layout.addWidget(QLabel("ROI Height:"), 3, 0)
        self.roi_h_spin = QSpinBox()
        self.roi_h_spin.setRange(1, 5000)
        self.roi_h_spin.setValue(self.roi_h)
        roi_layout.addWidget(self.roi_h_spin, 3, 1)

        self.apply_roi_button = QPushButton("Apply ROI")
        self.apply_roi_button.clicked.connect(self.apply_roi)
        control_layout.addWidget(self.apply_roi_button)

        # Timer to update the video feed.
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # roughly 30 ms between frames

    def apply_roi(self):
        # Get ROI values from the spin boxes and update the camera ROI.
        self.roi_x = int(self.roi_x_spin.value())
        self.roi_y = int(self.roi_y_spin.value())
        self.roi_w = int(self.roi_w_spin.value())
        self.roi_h = int(self.roi_h_spin.value())
        self.camera.CROI((self.roi_x, self.roi_y, self.roi_w, self.roi_h))
        print(f"Applied ROI: {(self.roi_x, self.roi_y, self.roi_w, self.roi_h)}")

    def auto_adjust_exposure(self, image, target_ratio=0.8):
        if image.dtype == np.uint16:
            sat_value = 65535
        else:
            sat_value = 255
        target = target_ratio * sat_value
        current_max = np.max(image)
        try:
            current_exposure = self.camera.ExposureTime  # using property directly
        except AttributeError:
            current_exposure = 1.0
        factor = target / current_max if current_max > 0 else 1.0
        calculated_exposure = current_exposure * factor
        # Gentle update: only 1% of the new value is added each time.
        new_exposure = 0.99 * current_exposure + 0.01 * calculated_exposure
        self.camera.ExposureTime = max(
            self._least_exposure, min(self._max_exposure, new_exposure)
        )
        return new_exposure

    def update_frame(self):
        # Grab an image from the camera.
        im = self.camera.grab_image()
        # Convert 16-bit images to 8-bit for display if needed.
        if im.dtype == np.uint16:
            im_disp = cv2.convertScaleAbs(im, alpha=(255.0 / 65535))
        else:
            im_disp = im.copy()

        # Use auto exposure if checked, otherwise set manual exposure/gain.
        if self.auto_exposure_checkbox.isChecked():
            self.auto_adjust_exposure(im)
        else:
            self.camera.ExposureTime = self.exposure_spin.value()
            self.camera.Gain = self.gain_spin.value()

        # Process the image: detect beam spots and overlay Gaussian fits.
        minThresh = self.blob_min_threshold_spin.value()
        maxThresh = self.blob_max_threshold_spin.value()
        spots = blob_detector(im, minThreshold=minThresh, maxThreshold=maxThresh)
        self.current_spots = spots  # store for mouse selection

        # spots = blob_detector(im)
        # Convert grayscale to BGR for colored overlays.
        disp = cv2.cvtColor(im_disp, cv2.COLOR_GRAY2BGR)
        for spot in self.current_spots:
            # Get centroid and sigma parameters.
            x = int(round(spot["x"]))
            y = int(round(spot["y"]))
            sigma0 = spot["sigma_0"]
            sigma1 = spot["sigma_1"]
            vec0 = spot["vec_0"]
            # Calculate ellipse orientation from the first eigenvector.
            angle = np.degrees(np.arctan2(vec0[1], vec0[0]))
            axes = (int(round(sigma0)), int(round(sigma1)))
            # Draw the ellipse for the beam spot.
            cv2.ellipse(disp, (x, y), axes, angle, 0, 360, (0, 255, 0), 2)

            # Calculate endpoints for the sigma axes.
            angle_rad = np.radians(angle)
            x_major = int(round(x + sigma0 * np.cos(angle_rad)))
            y_major = int(round(y + sigma0 * np.sin(angle_rad)))
            x_minor = int(round(x + sigma1 * np.cos(angle_rad + np.pi / 2)))
            y_minor = int(round(y + sigma1 * np.sin(angle_rad + np.pi / 2)))

            # Draw lines representing the sigma axes.
            cv2.line(
                disp, (x, y), (x_major, y_major), (255, 0, 0), 2
            )  # major axis in blue
            cv2.line(
                disp, (x, y), (x_minor, y_minor), (0, 0, 255), 2
            )  # minor axis in red

            # Overlay TeX-style labels for sigma.
            cv2.putText(
                disp,
                "0",
                (x_major, y_major),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1,
            )
            cv2.putText(
                disp,
                "1",
                (x_minor, y_minor),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
            )

            # Overlay xy coordinate and sigma values (using TeX-like formatting) on the image.
            cv2.putText(
                disp,
                f"(x: {x}, y: {y})",
                (x + 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )
            cv2.putText(
                disp,
                f"s0: {sigma0:.1f}, s1: {sigma1:.1f}",
                (x + 10, y + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )
        # Draw fixed blobs (with, say, yellow outlines and a label)
        for spot in self.fixed_blobs:
            x = int(round(spot["x"]))
            y = int(round(spot["y"]))
            sigma0 = spot["sigma_0"]
            sigma1 = spot["sigma_1"]
            vec0 = spot["vec_0"]
            angle = np.degrees(np.arctan2(vec0[1], vec0[0]))
            axes = (int(round(sigma0)), int(round(sigma1)))
            cv2.ellipse(disp, (x, y), axes, angle, 0, 360, (0, 255, 255), 2)
            cv2.putText(
                disp,
                "FIXED",
                (x + 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
            )

            # Also print the information to the console.
            # print(f"Spot at ({x}, {y}) with sigma0: {sigma0:.1f}, sigma1: {sigma1:.1f}")

        # Convert the processed image to QImage format and display it.
        height, width, channel = disp.shape
        self.last_image_size = (disp.shape[1], disp.shape[0])

        bytesPerLine = 3 * width
        qImg = QImage(disp.data, width, height, bytesPerLine, QImage.Format_BGR888)
        self.video_label.setPixmap(QPixmap.fromImage(qImg))

    def handle_mouse_press(self, event):
        pos = event.pos()
        # Ensure we know the image dimensions (store them in update_frame)
        if not hasattr(self, "last_image_size"):
            return
        label_w = self.video_label.width()
        label_h = self.video_label.height()
        img_w, img_h = self.last_image_size  # set in update_frame()
        scale_x = img_w / label_w
        scale_y = img_h / label_h
        click_x = pos.x() * scale_x
        click_y = pos.y() * scale_y

        # Check if the click is near a fixed blob (toggle removal if so)
        removal_indices = []
        for i, spot in enumerate(self.fixed_blobs):
            x = spot["x"]
            y = spot["y"]
            if np.hypot(click_x - x, click_y - y) < 20:
                removal_indices.append(i)
        if removal_indices:
            # Remove any fixed blobs within the threshold and exit.
            for idx in sorted(removal_indices, reverse=True):
                del self.fixed_blobs[idx]
            return

        # Otherwise, check current detected spots to add as fixed
        if hasattr(self, "current_spots"):
            closest_spot = None
            min_dist = float("inf")
            for spot in self.current_spots:
                x = spot["x"]
                y = spot["y"]
                dist = np.hypot(click_x - x, click_y - y)
                if dist < min_dist and dist < 20:
                    min_dist = dist
                    closest_spot = spot
            if closest_spot is not None:
                self.fixed_blobs.append(closest_spot)

    def closeEvent(self, event):
        # Clean up by closing the camera when the window is closed.
        self.camera.close()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = BeamProfilerGUI()
    window.show()
    sys.exit(app.exec_())
