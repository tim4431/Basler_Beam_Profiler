import numpy as np, matplotlib.pyplot as plt
from pypylon import pylon
import time


class BaslerCamera:
    def __init__(self):
        self.camera = pylon.InstantCamera(
            pylon.TlFactory.GetInstance().CreateFirstDevice()
        )
        self.camera.Open()
        self.camera.PixelFormat.Value = "Mono12"
        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        self.Gain = 0
        self.ExposureTime = 200
        self.converter = pylon.ImageFormatConverter()
        self.converter.OutputPixelFormat = (
            pylon.PixelType_Mono16
        )  # wrapping Mono12 data in Mono16, 16x value
        self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

    def pixel_to_coord(self, i, j):
        x = (i - self.W / 2) * self.pixel_size
        y = (j - self.H / 2) * self.pixel_size
        return x, y

    @property
    def Gain(self):
        return self.camera.Gain.Value

    @Gain.setter
    def Gain(self, value):
        self.camera.Gain.Value = value

    @property
    def ExposureTime(self):
        return self.camera.ExposureTime.Value

    @ExposureTime.setter
    def ExposureTime(self, value):
        self.camera.ExposureTime.Value = value

    @property
    def ROI(self):
        """
        Get the current ROI as (OffsetX, OffsetY, Width, Height)
        """
        return (
            self.camera.OffsetX.Value,
            self.camera.OffsetY.Value,
            self.camera.Width.Value,
            self.camera.Height.Value,
        )

    @ROI.setter
    def ROI(self, value):
        """
        Set the ROI as (OffsetX, OffsetY, Width, Height)
        """
        offset_x, offset_y, width, height = value

        # Ensure ROI does not exceed sensor size
        max_width = self.camera.WidthMax.Value
        max_height = self.camera.HeightMax.Value

        if width > max_width:
            width = max_width
        if height > max_height:
            height = max_height

        if offset_x + width > max_width:
            offset_x = max_width - width
        if offset_y + height > max_height:
            offset_y = max_height - height

        # Stop grabbing before changing ROI
        was_grabbing = self.camera.IsGrabbing()
        if was_grabbing:
            self.camera.StopGrabbing()

        # Apply ROI
        self.camera.Width.Value = int(width)
        self.camera.Height.Value = int(height)
        self.camera.OffsetX.Value = int(offset_x)
        self.camera.OffsetY.Value = int(offset_y)
        # Restart grabbing
        if was_grabbing:
            self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    def CROI(self, value):
        # (cx,cy,w,h)
        cx, cy, w, h = value
        cx = self.W / 2 + cx
        cy = self.H / 2 + cy
        self.ROI = (int(cx - w / 2), int(cy - h / 2), w, h)

    def grab_image(self):
        grab_result = self.camera.RetrieveResult(
            1000, pylon.TimeoutHandling_ThrowException
        )
        if grab_result.GrabSucceeded():
            image = self.converter.Convert(grab_result)
            img = np.array(image.GetArray(), dtype=np.uint16)
            grab_result.Release()
            return img
        return None

    def close(self):
        self.camera.StopGrabbing()
        self.camera.Close()


class BaslerC(BaslerCamera):
    def __init__(self):
        super().__init__()
        self.W = 4024
        self.H = 3036
        self.pixel_size = 3.45e-6


class Baslera2A5060(BaslerCamera):
    def __init__(self):
        super().__init__()
        self.W = 5060
        self.H = 5060
        self.pixel_size = 2.5e-6


class Baslera2A4504(BaslerCamera):
    def __init__(self):
        super().__init__()
        self.W = 4504
        self.H = 4504
        self.pixel_size = 2.74e-6


if __name__ == "__main__":
    camera = Baslera2A4504()
    camera.CROI((0, 0, 2000, 2000))
    img = camera.grab_image()
    camera.close()
    plt.imshow(img, cmap="gray")
    plt.colorbar()
    plt.show()
