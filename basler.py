import numpy as np, matplotlib.pyplot as plt
from pypylon import pylon
import time, logging
from collections import deque


class BaslerCamera:
    def __init__(self, mode="8Bit"):
        self.mode = mode
        self.auto_exposure = False
        self.MIN_EXPOSURE = 30
        self.MAX_EXPOSURE = 20000
        self._ae_buffer = deque(maxlen=5)
        self.pid_kp = 0.2  # Proportional gain; tune as needed
        self.pid_ki = 0  # Integral gain; tune as needed
        self.pid_kd = 0.1  # Derivative gain; set to 0 for PI control if desired
        self.pid_integral = 0.0
        self.pid_last_error = 0.0
        #
        self.camera = pylon.InstantCamera(
            pylon.TlFactory.GetInstance().CreateFirstDevice()
        )
        logging.info(
            f"Connected to camera: {self.camera.GetDeviceInfo().GetModelName()}"
        )
        #
        try:
            self.camera.Open()
            self.W = int(self.camera.Width.Max)
            self.H = int(self.camera.Height.Max)
        except Exception as e:
            logging.error(f"Failed to open camera: {e}")
            return
        # Set camera pixel format based on mode:
        self.converter = pylon.ImageFormatConverter()
        if mode == "16Bit":
            logging.info("Camera operating in 12-bit mode")
            self.camera.PixelFormat.Value = "Mono12"
            self.converter.OutputPixelFormat = (
                pylon.PixelType_Mono16
            )  # Wrap Mono12 in 16-bit
        elif mode == "8Bit":
            logging.info("Camera operating in 8-bit mode")
            self.camera.PixelFormat.Value = "Mono8"
            self.converter.OutputPixelFormat = (
                pylon.PixelType_Mono8
            )  # Traditional 8-bit for fast rendering
        else:
            raise ValueError(f"Invalid mode: {mode}")
        self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
        #
        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        self.Gain = 0
        self.ExposureTime = 200

        # Set converter output based on mode:

    def pixel_to_coord(self, i, j):
        x = (i - self.W / 2) * self.pixel_size
        y = (j - self.H / 2) * self.pixel_size
        return x, y

    @property
    def Gain(self):
        return self.camera.Gain.Value

    @Gain.setter
    def Gain(self, value):
        logging.debug(f"Gain set to: {self.camera.Gain.Value}")
        self.camera.Gain.Value = value

    @property
    def ExposureTime(self):
        return self.camera.ExposureTime.Value

    @ExposureTime.setter
    def ExposureTime(self, value):
        value = max(self.MIN_EXPOSURE, min(value, self.MAX_EXPOSURE))
        logging.debug(f"Exposure time set to: {self.camera.ExposureTime.Value}")
        self.camera.ExposureTime.Value = value

    @property
    def ROI(self):
        """
        Get the current ROI as (Width, Height, OffsetX, OffsetY).
        """
        return (
            self.camera.Width.Value,
            self.camera.Height.Value,
            self.camera.OffsetX.Value,
            self.camera.OffsetY.Value,
        )

    def _set_ROI(self, width, height, offset_x, offset_y):
        # Ensure ROI does not exceed sensor size.
        width = max(10, min(width, self.W))
        height = max(10, min(height, self.H))
        offset_x = min(offset_x, self.W - width)
        offset_y = min(offset_y, self.H - height)

        # logging.info(f"Calc ROI to: {width}, {height}, {offset_x}, {offset_y}")

        # Align values to allowed increments
        width = self._align_value(width, self.camera.Width)
        height = self._align_value(height, self.camera.Height)
        offset_x = self._align_value(offset_x, self.camera.OffsetX)
        offset_y = self._align_value(offset_y, self.camera.OffsetY)

        # logging.info(f"Set ROI to: {width}, {height}, {offset_x}, {offset_y}")

        self.camera.Width.Value = int(width)
        self.camera.Height.Value = int(height)
        self.camera.OffsetX.Value = int(offset_x)
        self.camera.OffsetY.Value = int(offset_y)

    def _set_autofunc_ROI(self, width, height, offset_x, offset_y):
        self.camera.AutoFunctionROIWidth.Value = int(width)
        self.camera.AutoFunctionROIHeight.Value = int(height)
        self.camera.AutoFunctionROIOffsetX.Value = int(offset_x)
        self.camera.AutoFunctionROIOffsetY.Value = int(offset_y)

        logging.info(f"Set AutoFunc ROI to: {width}, {height}, {offset_x}, {offset_y}")

    @ROI.setter
    def ROI(self, value):
        """
        Set the ROI as (Width, Height, OffsetX, OffsetY), aligning values to allowed increments.
        """
        if not self.camera.IsOpen():
            logging.error("Camera is not open, cannot set ROI.")
            return

        width, height, offset_x, offset_y = value

        try:
            # Stop grabbing before changing ROI.
            was_grabbing = self.camera.IsGrabbing()
            if was_grabbing:
                self.camera.StopGrabbing()

            # Apply ROI in a recommended order.
            self._set_ROI(width, height, offset_x, offset_y)
            # logging.info(f"ROI set to: {self.ROI}")

            # Restart grabbing if it was running.
            if was_grabbing:
                self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        except Exception as e:
            logging.error(f"Failed to set ROI: {e}")

    def _align_value(self, value, node):
        """
        Aligns 'value' down to the nearest value that is valid for the given node,
        based on the node's minimum value and increment.
        """
        min_val = node.Min
        inc_val = node.Inc
        # Calculate remainder to see how much value exceeds a valid step.
        remainder = (value - min_val) % inc_val
        if remainder != 0:
            value -= remainder  # floor to nearest valid increment.
        return value

    def set_default_roi(self):
        """
        Sets the ROI to the full sensor area.
        """
        self.ROI = self.default_roi

    def CROI(self, value):
        # (cx,cy,w,h) where (cx,cy) is relative to center.
        cx, cy, w, h = value
        cx = self.W / 2 + cx
        cy = self.H / 2 + cy
        self.ROI = (w, h, int(cx - w / 2), int(cy - h / 2))

    # def AutoExposureOn(self):
    #     minLowerLimit = 30
    #     maxUpperLimit = 20000
    #     self.camera.AutoExposureTimeLowerLimit.Value = minLowerLimit
    #     self.camera.AutoExposureTimeUpperLimit.Value = maxUpperLimit
    #     # Set the target brightness value to 0.6
    #     self.camera.AutoTargetBrightness.Value = 0.6
    #     # Select auto function ROI 1
    #     self.camera.AutoFunctionROISelector.Value = "ROI1"
    #     width, height, offset_x, offset_y = self.ROI
    #     self._set_autofunc_ROI(width, height, offset_x, offset_y)
    #     # Enable the 'Brightness' auto function (Gain Auto + Exposure Auto)
    #     # for the auto function ROI selected
    #     self.camera.AutoFunctionROIUseBrightness.Value = True
    #     # Enable Exposure Auto by setting the operating mode to Continuous
    #     self.camera.ExposureAuto.Value = "Continuous"

    # def AutoExposureOff(self):
    #     self.camera.ExposureAuto.Value = "Off"

    def AutoExposureOn(self, range=None):
        self.auto_exposure = True
        if range is not None:
            self.MIN_EXPOSURE, self.MAX_EXPOSURE = range

    def AutoExposureOff(self):
        self.auto_exposure = False

    def grab_image(self):
        grab_result = self.camera.RetrieveResult(
            1000, pylon.TimeoutHandling_ThrowException
        )
        if grab_result.GrabSucceeded():
            image = self.converter.Convert(grab_result)
            img = np.array(
                image.GetArray(),
                dtype=(np.uint8 if self.mode == "8Bit" else np.uint16),
            )
            grab_result.Release()

            # Hand-written auto-exposure logic:
            if self.auto_exposure:
                # Determine saturation and target intensity based on mode.
                saturation = 255 if self.mode == "8Bit" else 65535
                target = 0.9 * saturation
                tolerance = 0.1 * target
                max_val = np.max(img)
                self._ae_buffer.append(max_val)
                current_exposure = self.ExposureTime
                EXPOSURE_STEP = 0.1

                if max_val < target - tolerance:
                    # Image too dark: increase exposure and reset PID state.
                    self.ExposureTime = current_exposure * (1 + EXPOSURE_STEP)
                    self.pid_integral = 0.0
                    self.pid_last_error = 0.0
                elif max_val > target + tolerance:
                    # Image too bright: decrease exposure and reset PID state.
                    self.ExposureTime = current_exposure * (1 - EXPOSURE_STEP)
                    self.pid_integral = 0.0
                    self.pid_last_error = 0.0
                else:
                    # Within tolerance: use PID for fine adjustment.
                    val = np.mean(self._ae_buffer)
                    error = (target - val) / saturation
                    self.pid_integral += error  # Optionally multiply by dt if available
                    derivative = (
                        error - self.pid_last_error
                    )  # Optionally divide by dt if available
                    self.pid_last_error = error

                    # Compute the control adjustment.
                    control = (
                        self.pid_kp * error
                        + self.pid_ki * self.pid_integral
                        + self.pid_kd * derivative
                    )

                    new_exposure = current_exposure * (1 + control)
                    new_exposure = max(
                        self.MIN_EXPOSURE, min(new_exposure, self.MAX_EXPOSURE)
                    )
                    self.ExposureTime = new_exposure

            return img

        return None

    def close(self):
        self.camera.StopGrabbing()
        self.camera.Close()


class Baslera2A5060(BaslerCamera):
    def __init__(self, mode):
        super().__init__(mode)
        self.default_roi = (5060, 5060, 4, 4)
        self.pixel_size = 2.5e-6


class Baslera2A4504(BaslerCamera):
    def __init__(self, mode):
        super().__init__(mode)
        self.default_roi = (4504, 4504, 4, 4)
        self.pixel_size = 2.74e-6


if __name__ == "__main__":
    camera = Baslera2A4504(mode="16Bit")
    # Example: set a custom ROI
    # camera.CROI((0, 0, 4000, 4000))
    img = camera.grab_image()
    # Optionally: reset to full ROI
    # camera.set_full_roi()
    camera.close()
    plt.imshow(img, cmap="gray")
    plt.colorbar()
    # plt.savefig("test.png")
    plt.show()
