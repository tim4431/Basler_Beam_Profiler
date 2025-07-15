import cv2, sys, time, logging, math, os, tkinter as tk
import numpy as np
from tkinter import filedialog

# ------------------------------ CONFIG ---------------------------------- #

MIN_EXPOSURE = 30  # µs
MAX_EXPOSURE = 100000  # µs
FPS_LIMIT = 60  # frames / s (UI update)
WINDOW_SIZE = 1100  # longest display edge in px
PAD_COLOR = (128, 128, 128)  # gray padding
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# ------------------------------ LOGGING --------------------------------- #

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# ------------------------------ CAMERA ---------------------------------- #

# sys.path.append("../../src")  # project‑local libs
from basler import BaslerCamera  # noqa: E402
from blob_detector import blob_detector, render_blobs_with_img  # noqa: E402
from pypylon import pylon

factory = pylon.TlFactory.GetInstance()
available = factory.EnumerateDevices()
cameras = [BaslerCamera(mode="8Bit", device_idx=i) for i, _ in enumerate(available)]

if not cameras:
    logging.error("No Basler cameras detected – aborting.")
    sys.exit(1)

# cache per‑camera state so switching is seamless
cached_state: dict[str, dict] = {}

# ------------------------------ ROI MODEL ------------------------------- #


class ROIModel:
    """Maintain ROI as (scale, aspect, centre) → tuple understood by Basler."""

    def __init__(self, sensor_w: int, sensor_h: int):
        self.full_w = sensor_w
        self.full_h = sensor_h
        self.scale = 1.0  # fraction of *width*
        self.aspect = sensor_w / sensor_h  # w / h  (initial native)
        self.cx = sensor_w / 2  # centre in pixel coords
        self.cy = sensor_h / 2

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    @property
    def size(self):
        w = max(1, int(self.full_w * self.scale))
        h = max(1, int(w / self.aspect))
        # shrink if we spill vertically
        if h > self.full_h:
            h = self.full_h
            w = int(h * self.aspect)
        return w, h

    @property
    def tuple(self):
        """Return (w, h, offset_x, offset_y) as expected by Basler."""
        w, h = self.size
        ox = int(self.cx - w / 2)
        oy = int(self.cy - h / 2)
        # clamp
        ox = max(0, min(ox, self.full_w - w))
        oy = max(0, min(oy, self.full_h - h))
        # recalc centre to honour clamping
        self.cx = ox + w / 2
        self.cy = oy + h / 2
        return w, h, ox, oy

    def display_to_sensor(self, disp_x, disp_y, display_size, pad_l, pad_t):
        w_roi, h_roi, ox, oy = self.tuple
        # remove padding
        x_no_pad = disp_x - pad_l
        y_no_pad = disp_y - pad_t
        # get real image size
        scale = display_size / max(w_roi, h_roi)
        view_w = int(w_roi * scale)
        view_h = int(h_roi * scale)
        # convert to relative coordinates
        r_x = x_no_pad / view_w if view_w > 0 else 0
        r_y = y_no_pad / view_h if view_h > 0 else 0
        # convert to sensor coordinates
        sensor_x = ox + r_x * w_roi
        sensor_y = oy + r_y * h_roi
        return sensor_x, sensor_y

    def sensor_to_display(self, sensor_x, sensor_y, display_size, pad_l, pad_t):
        w_roi, h_roi, ox, oy = self.tuple
        #  convert to relative coordinates
        r_x = (sensor_x - ox) / w_roi if w_roi > 0 else 0
        r_y = (sensor_y - oy) / h_roi if h_roi > 0 else 0
        # get real image size
        scale = display_size / max(w_roi, h_roi)
        view_w = int(w_roi * scale)
        view_h = int(h_roi * scale)
        # convert to display coordinates
        disp_x = int(r_x * view_w) + pad_l
        disp_y = int(r_y * view_h) + pad_t
        return disp_x, disp_y

    def coord_in_roi_to_sensor(self, x, y):
        """Convert coordinates in the ROI to sensor coordinates."""
        w_roi, h_roi, ox, oy = self.tuple
        # convert to relative coordinates
        r_x = x / w_roi if w_roi > 0 else 0
        r_y = y / h_roi if h_roi > 0 else 0
        # convert to sensor coordinates
        sensor_x = ox + r_x * w_roi
        sensor_y = oy + r_y * h_roi
        return sensor_x, sensor_y

    # -------------------------- interactions --------------------------- #
    def keep_point_fixed(
        self, sensor_x: float, sensor_y: float, new_scale=None, new_aspect=None
    ):
        """Update scale/aspect so that *sensor_x, sensor_y* remain
        at the *same absolute position* in the sensor afterwards.
        Additionally, when changing *aspect*, keep the longest edge length
        unchanged so zoom level feels intuitive (requirement #4)."""
        # Current ROI geometry
        w_old, h_old, ox_old, oy_old = self.tuple
        long_old = max(w_old, h_old)
        r_x = (sensor_x - ox_old) / w_old if w_old else 0.5
        r_y = (sensor_y - oy_old) / h_old if h_old else 0.5

        # First apply new aspect so we can compensate scale later
        if new_aspect is not None:
            new_aspect = max(0.1, min(new_aspect, 10.0))
            # Predict size with *current* scale
            w_tmp = self.full_w * self.scale
            h_tmp = w_tmp / new_aspect
            long_tmp = max(w_tmp, h_tmp)
            if long_tmp > 0:
                self.scale *= long_old / long_tmp  # keep longest edge constant
            self.aspect = new_aspect

        # Apply zoom afterwards in case both happen in same event
        if new_scale is not None:
            self.scale = max(0.1, min(new_scale, 1.0))

        # Re‑compute ROI and recalc centre while clamping
        w_new, h_new = self.size
        ox_new = sensor_x - r_x * w_new
        oy_new = sensor_y - r_y * h_new
        self.cx = np.clip(ox_new + w_new / 2, w_new / 2, self.full_w - w_new / 2)
        self.cy = np.clip(oy_new + h_new / 2, h_new / 2, self.full_h - h_new / 2)


# --------------------------- GUI HELPERS ------------------------------- #


def choose_filename():
    root = tk.Tk()
    root.withdraw()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    default = f"vipa_{timestamp}.jpg"
    fp = filedialog.asksaveasfilename(
        initialdir=DATA_DIR,
        initialfile=default,
        title="Save Image",
        filetypes=(("JPEG files", "*.jpg"), ("All files", "*.*")),
    )
    root.destroy()
    return fp


# --------------------------- STATISTICS ------------------------------- #


def classify_dots_grid(spots, eps=None, min_samples=2):
    """
    使用基于邻近距离分析的聚类算法对点阵进行行列分类

    参数:
        spots: 包含点信息的列表，每个点需要有'x'和'y'键
        eps: 相邻聚类的最大距离，None表示自动确定
        min_samples: 形成一个有效行/列所需的最小点数

    返回:
        rows: 按行分类的点列表
        columns: 按列分类的点列表
        stats: 包含行列统计信息的字典
    """
    if not spots:
        return [], [], {}

    # 提取坐标
    xs = np.array([s["x"] for s in spots])
    ys = np.array([s["y"] for s in spots])

    # ------------------------- 基于距离分析的行列聚类 -------------------------

    # 分析X方向（查找列）
    sorted_x_indices = np.argsort(xs)
    sorted_xs = xs[sorted_x_indices]

    # 计算相邻X坐标差值
    x_gaps = np.diff(sorted_xs)

    # 自动确定聚类阈值
    if eps is None:
        # 使用基于分布的自适应阈值
        if len(x_gaps) > 0:
            median_x_gap = np.median(x_gaps)
            # Tukey方法识别异常大间隔
            q75, q25 = np.percentile(x_gaps, [85, 15])
            iqr = q75 - q25
            x_threshold = q75 + (70 / 15) * iqr

            # 使用最小阈值确保鲁棒性
            x_threshold = max(x_threshold, 2.0 * median_x_gap)
        else:
            x_threshold = 1.0  # 默认值
    else:
        x_threshold = eps

    # 基于间隔识别列边界
    x_break_points = np.where(x_gaps > x_threshold)[0]

    # 根据边界分配列
    columns = []
    start_idx = 0

    for break_point in x_break_points:
        end_idx = break_point + 1
        if end_idx - start_idx >= min_samples:
            col_indices = sorted_x_indices[start_idx:end_idx]
            col_points = [spots[i] for i in col_indices]
            columns.append(col_points)
        start_idx = end_idx

    # 处理最后一列
    if len(sorted_xs) - start_idx >= min_samples:
        col_indices = sorted_x_indices[start_idx:]
        col_points = [spots[i] for i in col_indices]
        columns.append(col_points)

    # 分析Y方向（查找行）
    sorted_y_indices = np.argsort(ys)
    sorted_ys = ys[sorted_y_indices]

    # 计算相邻Y坐标差值
    y_gaps = np.diff(sorted_ys)

    # 自动确定聚类阈值
    if eps is None:
        # 使用基于分布的自适应阈值
        if len(y_gaps) > 0:
            median_y_gap = np.median(y_gaps)
            # Tukey方法识别异常大间隔````````````````````````````````````````````````````````````````````````````````````1  `222222222222222222222222222222222222222222222222222222```````````````````````````````````````````````````` `111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111      `111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111`                                                                                                                                                                                                                                                                                                                                                                                                                                                 ``
            q75, q25 = np.percentile(y_gaps, [85, 15])
            iqr = q75 - q25
            y_threshold = q75 + (70 / 15) * iqr

            # 使用最小阈值确保鲁棒性
            y_threshold = max(y_threshold, 2.0 * median_y_gap)
        else:
            y_threshold = 1.0  # 默认值
    else:
        y_threshold = eps

    # 基于间隔识别行边界
    y_break_points = np.where(y_gaps > y_threshold)[0]

    # 根据边界分配行
    rows = []
    start_idx = 0

    for break_point in y_break_points:
        end_idx = break_point + 1
        if end_idx - start_idx >= min_samples:
            row_indices = sorted_y_indices[start_idx:end_idx]
            row_points = [spots[i] for i in row_indices]
            rows.append(row_points)
        start_idx = end_idx

    # 处理最后一行
    if len(sorted_ys) - start_idx >= min_samples:
        row_indices = sorted_y_indices[start_idx:]
        row_points = [spots[i] for i in row_indices]
        rows.append(row_points)

    # ----------------------- 对每行和每列内点进行排序 --------------------------
    # 每行内按x坐标排序
    for row in rows:
        row.sort(key=lambda s: s["x"])

    # 每列内按y坐标排序
    for col in columns:
        col.sort(key=lambda s: s["y"])

    # -------------------------- 计算统计信息 -----------------------------
    stats = {
        "rows": {"mean_dx": [], "std_x": [], "std_y": []},
        "columns": {"mean_dy": [], "std_y": [], "std_x": []},
    }

    # 计算每行的统计信息
    for row in rows:
        if len(row) >= 2:
            row_xs = np.array([s["x"] for s in row])
            row_xs.sort()
            row_dxs = np.diff(row_xs)

            stats["rows"]["mean_dx"].append(float(row_dxs.mean()))
            stats["rows"]["std_x"].append(float(row_dxs.std()))

            # calculate the deviation in y for each row
            row_ys = np.array([s["y"] for s in row])
            stats["rows"]["std_y"].append(float(row_ys.std()))

    # 计算每列的统计信息
    for col in columns:
        if len(col) >= 2:
            col_ys = np.array([s["y"] for s in col])
            col_ys.sort()
            col_dys = np.diff(col_ys)

            stats["columns"]["mean_dy"].append(float(col_dys.mean()))
            stats["columns"]["std_y"].append(float(col_dys.std()))

            # calculate the deviation in x for each column
            col_xs = np.array([s["x"] for s in col])
            stats["columns"]["std_x"].append(float(col_xs.std()))
    # ----------------------- 计算平均统计信息 -----------------------------

    # 计算所有行和列的平均统计信息
    for key in ["mean_dx", "std_x", "std_y"]:
        if stats["rows"][key]:
            stats["rows"][f"avg_{key}"] = float(np.mean(stats["rows"][key]))

    for key in ["mean_dy", "std_y", "std_x"]:
        if stats["columns"][key]:
            stats["columns"][f"avg_{key}"] = float(np.mean(stats["columns"][key]))

    return rows, columns, stats


# --------------------------- APPLICATION ------------------------------- #


class BaslerViewer:
    def __init__(self):
        self.curr_idx = 0
        self.camera = cameras[self.curr_idx]
        self.roi_model = ROIModel(self.camera.W, self.camera.H)
        self.camera.ROI = self.roi_model.tuple
        self._auto_exp = False
        self.do_fitting = False
        self.show_stats = False  # toggled by 'g'
        self.row_col_fitting = False  # toggled by 'h'
        self.last_mouse = (0, 0)  # sensor coords
        self.rect_disp = None  # generic display rectangle (x1,y1,x2,y2)
        self.rect_sensor = None  # sensor coords
        self.fit_rect_disp = None  # fitting‑region rectangle (display coords)
        self.fit_rect_sensor = None  # fitting‑region rectangle (sensor coords)
        self.exposure_us = 200  # default, in us
        self.avg_dt = 1 / FPS_LIMIT
        self._init_window()

    @property
    def auto_exp(self):
        return self._auto_exp

    @auto_exp.setter
    def auto_exp(self, value):
        self._auto_exp = value
        if value:
            self.camera.AutoExposureOn(range=(MIN_EXPOSURE, MAX_EXPOSURE))
        else:
            self.camera.AutoExposureOff()
            self.exposure_us = int(
                np.clip(self.exposure_us, MIN_EXPOSURE, MAX_EXPOSURE)
            )

    # ---------------------- window & callbacks ---------------------- #
    def _init_window(self):
        cv2.namedWindow("Basler", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Basler", WINDOW_SIZE, WINDOW_SIZE)
        cv2.setMouseCallback("Basler", self.on_mouse)

    def on_mouse(self, event, x, y, flags, _param):
        # calculate padding
        w_roi, h_roi, ox, oy = self.camera.ROI
        scale = WINDOW_SIZE / max(w_roi, h_roi)
        view_w = int(w_roi * scale)
        view_h = int(h_roi * scale)
        pad_l = (WINDOW_SIZE - view_w) // 2
        pad_t = (WINDOW_SIZE - view_h) // 2

        # check if mouse is in the image area
        if (pad_l <= x < pad_l + view_w) and (pad_t <= y < pad_t + view_h):
            sensor_x, sensor_y = self.roi_model.display_to_sensor(
                x, y, WINDOW_SIZE, pad_l, pad_t
            )
            self.last_mouse = (int(sensor_x), int(sensor_y))
        else:
            # mouse is outside the image area
            pass

        ctrl_down = flags & cv2.EVENT_FLAG_CTRLKEY

        # ---------------------------------------------------------- scroll wheel
        if event == cv2.EVENT_MOUSEWHEEL:
            if ctrl_down:
                # change aspect ratio while keeping long edge fixed
                factor = 0.95 if flags > 0 else 1.05
                new_aspect = self.roi_model.aspect * factor
                self.roi_model.keep_point_fixed(
                    self.last_mouse[0], self.last_mouse[1], new_aspect=new_aspect
                )
            else:
                # zoom in/out
                delta = -0.05 if flags > 0 else 0.05
                new_scale = self.roi_model.scale + delta
                self.roi_model.keep_point_fixed(
                    self.last_mouse[0], self.last_mouse[1], new_scale=new_scale
                )
            self.camera.ROI = self.roi_model.tuple

        # ---------------------------------------------------------- rectangle draw
        if event == cv2.EVENT_LBUTTONDOWN:
            if (pad_l <= x < pad_l + view_w) and (pad_t <= y < pad_t + view_h):
                if ctrl_down:
                    self.fit_rect_disp = [x, y, x, y]
                else:
                    self.rect_disp = [x, y, x, y]

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.fit_rect_disp is not None:
                self.fit_rect_disp[2:] = [x, y]
            elif self.rect_disp is not None:
                self.rect_disp[2:] = [x, y]

        elif event == cv2.EVENT_LBUTTONUP:
            if self.fit_rect_disp is not None:
                x1, y1, x2, y2 = self.fit_rect_disp
                self.fit_rect_disp = None

                # convert the corners of the rectangle to sensor coordinates
                s_x1, s_y1 = self.roi_model.display_to_sensor(
                    min(x1, x2), min(y1, y2), WINDOW_SIZE, pad_l, pad_t
                )
                s_x2, s_y2 = self.roi_model.display_to_sensor(
                    max(x1, x2), max(y1, y2), WINDOW_SIZE, pad_l, pad_t
                )
                self.fit_rect_sensor = (int(s_x1), int(s_y1), int(s_x2), int(s_y2))

            elif self.rect_disp is not None:
                x1, y1, x2, y2 = self.rect_disp
                self.rect_disp = None

                # convert the corners of the rectangle to sensor coordinates
                s_x1, s_y1 = self.roi_model.display_to_sensor(
                    min(x1, x2), min(y1, y2), WINDOW_SIZE, pad_l, pad_t
                )
                s_x2, s_y2 = self.roi_model.display_to_sensor(
                    max(x1, x2), max(y1, y2), WINDOW_SIZE, pad_l, pad_t
                )
                self.rect_sensor = (int(s_x1), int(s_y1), int(s_x2), int(s_y2))

    # ----------------------------- main loop ---------------------------- #
    def run(self):
        while True:
            t0 = time.time()
            frame = self.camera.grab_image()
            if frame is None:
                continue
            if self.camera.mode == "16Bit":
                frame_disp = cv2.convertScaleAbs(frame, alpha=1 / 256.0)
            else:
                frame_disp = frame

            # ---------------------------------------------------- Blob detection
            spots = []
            if self.do_fitting:
                all_spots = blob_detector(frame)
                # restrict to fitting_rect if provided
                if self.fit_rect_sensor is not None:
                    x1, y1, x2, y2 = self.fit_rect_sensor
                    for s in all_spots:
                        s_x, s_y = self.roi_model.coord_in_roi_to_sensor(s["x"], s["y"])
                        if x1 <= s_x <= x2 and y1 <= s_y <= y2:
                            spots.append(s)
                else:
                    spots = all_spots

            # create RGB overlay image
            rgb = cv2.cvtColor(frame_disp, cv2.COLOR_GRAY2RGB)
            if self.do_fitting and spots:
                # 常规拟合模式
                if not self.row_col_fitting:
                    rgb = render_blobs_with_img(
                        rgb,
                        spots,
                        rgb=True,
                        render_axis=True,
                        render_xy=True,
                        render_sigma=True,
                        pixel_size=self.camera.pixel_size,
                    )

            # ------------------- resize & pad to square -------------- #
            w_roi, h_roi, *_ = self.camera.ROI
            scale = WINDOW_SIZE / max(w_roi, h_roi)
            view = cv2.resize(rgb, (int(w_roi * scale), int(h_roi * scale)))
            pad_l = (WINDOW_SIZE - view.shape[1]) // 2
            pad_r = WINDOW_SIZE - view.shape[1] - pad_l
            pad_t = (WINDOW_SIZE - view.shape[0]) // 2
            pad_b = WINDOW_SIZE - view.shape[0] - pad_t
            disp = cv2.copyMakeBorder(
                view, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=PAD_COLOR
            )

            # ------------------------- overlays ---------------------- #
            next_y = self._draw_hud(disp)
            self._calc_spot_statistics(spots)  # update stats
            self._draw_stats_bar(disp, next_y)  # if stats enabled draws
            self._draw_rectangles(disp, pad_l, pad_t)

            # 行列拟合可视化模式
            if self.row_col_fitting:
                self._draw_row_col_visualization(disp)

            cv2.imshow("Basler", disp)
            key = cv2.waitKeyEx(1)
            if not self._handle_key(key):
                break

            # fps regulation ---------------------------------------- #
            self._sleep_for_fps(time.time() - t0)

        self.camera.close()
        cv2.destroyAllWindows()

    # ------------------------------------------------------------------ #
    # HUD & overlays
    # ------------------------------------------------------------------ #
    def _draw_hud(self, img):
        """Draw HUD on top of gray padding (#2). Returns the y coordinate after
        the last HUD line so stats bar can follow."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        line_height = 26
        col = (255, 255, 255)
        HUD_X = 10
        HUD_Y = 5
        y = HUD_Y

        exposure_now = self.camera.ExposureTime
        if self.auto_exp:
            exposure_txt = f"Auto = {exposure_now:.0f}"
        else:
            exposure_txt = f"{exposure_now}"

        rect_w_um = (
            self.camera.pixel_size * (self.rect_sensor[2] - self.rect_sensor[0]) * 1e6
            if self.rect_sensor
            else 0
        )
        rect_h_um = (
            self.camera.pixel_size * (self.rect_sensor[3] - self.rect_sensor[1]) * 1e6
            if self.rect_sensor
            else 0
        )

        items = [
            f"Camera: {self.camera.model_name}, FPS: {1/self.avg_dt:4.1f}",
            f"ROI: {self.camera.ROI}  Mouse: {self.last_mouse}",
            f"Exposure: {exposure_txt} us",
            f"Rect: W = {rect_w_um:.1f} um, H = {rect_h_um:.1f} um",
            f"Fitting: {self.do_fitting}, Stats: {self.show_stats}, Row-Col: {self.row_col_fitting}",
        ]
        for txt in items:
            y += line_height
            cv2.putText(img, txt, (HUD_X, y), font, 0.7, col, 2, cv2.LINE_AA)
        return y  # next y for further drawings

    def _draw_rectangles(self, img, pad_l: int = 0, pad_t: int = 0):
        # get ROI size and scale
        w_roi, h_roi, ox, oy = self.camera.ROI
        scale = WINDOW_SIZE / max(w_roi, h_roi)

        # draw the ROI rectangle (when dragging)
        if self.rect_disp is not None:
            x1, y1, x2, y2 = self.rect_disp
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 2)

        if self.fit_rect_disp is not None:
            x1, y1, x2, y2 = self.fit_rect_disp
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # draw the ROI rectangle (when saved)
        if self.rect_sensor is not None:
            sx1, sy1, sx2, sy2 = self.rect_sensor
            p1_x, p1_y = self.roi_model.sensor_to_display(
                sx1, sy1, WINDOW_SIZE, pad_l, pad_t
            )
            p2_x, p2_y = self.roi_model.sensor_to_display(
                sx2, sy2, WINDOW_SIZE, pad_l, pad_t
            )
            cv2.rectangle(img, (p1_x, p1_y), (p2_x, p2_y), (255, 255, 255), 2)

        if self.fit_rect_sensor is not None:
            sx1, sy1, sx2, sy2 = self.fit_rect_sensor
            p1_x, p1_y = self.roi_model.sensor_to_display(
                sx1, sy1, WINDOW_SIZE, pad_l, pad_t
            )
            p2_x, p2_y = self.roi_model.sensor_to_display(
                sx2, sy2, WINDOW_SIZE, pad_l, pad_t
            )
            cv2.rectangle(img, (p1_x, p1_y), (p2_x, p2_y), (0, 255, 0), 2)

    def _calc_spot_statistics(self, spots):
        # 原始统计计算
        if len(spots) > 1:
            xs = np.array([s["x"] for s in spots])
            ys = np.array([s["y"] for s in spots])
            # argsort x
            xidx = np.argsort(xs)
            xs = xs[xidx]
            ys = ys[xidx]
            stats_dx = np.array([xs[i + 1] - xs[i] for i in range(len(xs) - 1)])
            stats_dy = np.array([ys[i + 1] - ys[i] for i in range(len(ys) - 1)])
            #
            sigma_0s = np.array([s["sigma_0"] for s in spots])
            sigma_1s = np.array([s["sigma_1"] for s in spots])
            stats_sigma = np.sqrt(sigma_0s * sigma_1s)

            # 将点分类为行和列
            rows, columns, grid_stats = classify_dots_grid(spots, eps=20)
        else:
            stats_dx = np.array([0.0])
            stats_dy = np.array([0.0])
            stats_sigma = np.array([0.0])
            rows, columns, grid_stats = [], [], {}

        self.stats_dx = stats_dx
        self.stats_dy = stats_dy
        self.stats_sigma = stats_sigma
        self.grid_stats = grid_stats
        self.rows = rows
        self.columns = columns

    def _draw_stats_bar(self, display_img, start_y):
        if not (self.show_stats):
            return

        pixel_to_um = self.camera.pixel_size * 1e6

        stats_dx = self.stats_dx
        stats_dy = self.stats_dy

        stats_std_dx = np.std(stats_dx)
        stats_std_dy = np.std(stats_dy)
        stats_dx_mean_um = np.mean(stats_dx) * self.camera.pixel_size * 1e6
        stats_dy_mean_um = np.mean(stats_dy) * self.camera.pixel_size * 1e6
        # calculate its second order statistics (curvature)
        # distinguishing plus and minus curvature
        if len(stats_dx) > 1:
            stats_dx_curvature = np.mean(np.diff(stats_dx, 1))
        else:
            stats_dx_curvature = 0.0
        if len(stats_dy) > 1:
            stats_dy_curvature = np.mean(np.diff(stats_dy, 1))
        else:
            stats_dy_curvature = 0.0

        if stats_std_dx is not None:
            bar_factor = 5  # scaling factor (adjust as needed)
            max_bar_length = 100  # maximum bar length for display
            line_height = 26
            bar_y = start_y + line_height  # start after HUD
            text_x = 10

            # 原始统计显示代码
            cv2.putText(
                display_img,
                f"std(x): {stats_std_dx:.2f}",
                (text_x, bar_y + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )
            bar_x = text_x + 160
            bar_length = int(min(stats_std_dx * bar_factor, max_bar_length))
            cv2.rectangle(
                display_img,
                (bar_x, bar_y),
                (bar_x + bar_length, bar_y + 20),
                (0, 255, 255),
                -1,
            )
            cv2.rectangle(
                display_img,
                (bar_x, bar_y),
                (bar_x + max_bar_length, bar_y + 20),
                (255, 255, 255),
                2,
            )

            text_x = bar_x + max_bar_length + 50
            cv2.putText(
                display_img,
                f"std(y): {stats_std_dy:.2f}",
                (text_x, bar_y + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )
            bar_x = text_x + 160
            bar_length = int(min(stats_std_dy * bar_factor, max_bar_length))
            cv2.rectangle(
                display_img,
                (bar_x, bar_y),
                (bar_x + bar_length, bar_y + 20),
                (0, 255, 255),
                -1,
            )
            cv2.rectangle(
                display_img,
                (bar_x, bar_y),
                (bar_x + max_bar_length, bar_y + 20),
                (255, 255, 255),
                2,
            )
            # 显示平均dx和dy值，以微米为单位
            text_mean_dx_x = bar_x + max_bar_length + 50
            cv2.putText(
                display_img,
                f"Dx: {stats_dx_mean_um:.1f} um",
                (text_mean_dx_x, bar_y + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )
            text_mean_dy_x = text_mean_dx_x + 200
            cv2.putText(
                display_img,
                f"Dy: {stats_dy_mean_um:.1f} um",
                (text_mean_dy_x, bar_y + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )

            if self.show_stats and not self.row_col_fitting:
                # display the curvature values in the next line, similar to std(x) and std(y)
                # the bar factor here is 2.5, bar display from -20 to 20
                bar_factor = 2.5  # scaling factor for curvature
                max_bar_length = 100  # maximum bar length for curvature display
                half_bar_length = int(max_bar_length / 2)
                next_line_y = bar_y + 30
                text_x = 10
                cv2.putText(
                    display_img,
                    f"CurX: {stats_dx_curvature:.2f}",
                    (text_x, next_line_y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                bar_x = text_x + 160
                bar_length = int(
                    min(abs(stats_dx_curvature) * bar_factor, half_bar_length)
                )

                cv2.rectangle(
                    display_img,
                    (bar_x, next_line_y),
                    (bar_x + max_bar_length, next_line_y + 20),
                    (255, 255, 255),
                    2,
                )
                if stats_dx_curvature < 0:  # from left to center
                    cv2.rectangle(
                        display_img,
                        (bar_x + half_bar_length - bar_length, next_line_y),
                        (bar_x + half_bar_length, next_line_y + 20),
                        (0, 0, 255),  # red for negative curvature
                        -1,
                    )
                else:
                    cv2.rectangle(
                        display_img,
                        (bar_x + half_bar_length, next_line_y),
                        (bar_x + half_bar_length + bar_length, next_line_y + 20),
                        (0, 255, 0),  # green for positive curvature
                        -1,
                    )
                text_x = bar_x + max_bar_length + 50
                cv2.putText(
                    display_img,
                    f"CurY: {stats_dy_curvature:.2f}",
                    (text_x, next_line_y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                bar_x = text_x + 160
                bar_length = int(
                    min(abs(stats_dy_curvature) * bar_factor, half_bar_length)
                )
                cv2.rectangle(
                    display_img,
                    (bar_x, next_line_y),
                    (bar_x + max_bar_length, next_line_y + 20),
                    (255, 255, 255),
                    2,
                )
                if stats_dy_curvature < 0:  # from top to center
                    cv2.rectangle(
                        display_img,
                        (bar_x + half_bar_length - bar_length, next_line_y),
                        (bar_x + half_bar_length, next_line_y + 20),
                        (0, 0, 255),  # red for negative curvature
                        -1,
                    )
                else:
                    cv2.rectangle(
                        display_img,
                        (bar_x + half_bar_length, next_line_y),
                        (bar_x + half_bar_length + bar_length, next_line_y + 20),
                        (0, 255, 0),  # green for positive curvature
                        -1,
                    )

                # bar for sigma
                stats_sigma = self.stats_sigma
                stats_sigma_std = np.std(stats_sigma)
                bar_x = text_x + 320
                bar_factor = 20  # scaling factor for curvature
                max_bar_length = 100  # maximum bar length for curvature display
                cv2.putText(
                    display_img,
                    f"std(s): {stats_sigma_std:.3f}",
                    (bar_x, next_line_y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                bar_x += 160
                bar_length = int(min(stats_sigma_std * bar_factor, max_bar_length))
                cv2.rectangle(
                    display_img,
                    (bar_x, next_line_y),
                    (bar_x + bar_length, next_line_y + 20),
                    (0, 255, 255),  # yellow for sigma std
                    -1,
                )
                cv2.rectangle(
                    display_img,
                    (bar_x, next_line_y),
                    (bar_x + max_bar_length, next_line_y + 20),
                    (255, 255, 255),  # white border
                    2,
                )
                # 显示平均sigma值，以像素为单位

        # ---- Row and column statistics ----

        grid_stats = self.grid_stats
        rows = self.rows
        columns = self.columns

        if self.row_col_fitting and grid_stats:
            for key in ["avg_mean_dx", "avg_std_x", "avg_std_y"]:
                if key in grid_stats["rows"]:
                    grid_stats["rows"][f"{key}_um"] = (
                        grid_stats["rows"][key] * pixel_to_um
                    )

            for key in ["avg_mean_dy", "avg_std_y", "avg_std_x"]:
                if key in grid_stats["columns"]:
                    grid_stats["columns"][f"{key}_um"] = (
                        grid_stats["columns"][key] * pixel_to_um
                    )

            # Row and column statistics
            next_line_y = bar_y + 40

            # Row statistics
            cv2.putText(
                display_img,
                f"NRow: {len(rows)}",
                (10, next_line_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )

            if "avg_mean_dx_um" in grid_stats["rows"]:
                cv2.putText(
                    display_img,
                    f"Dx: {grid_stats['rows']['avg_mean_dx_um']:.1f} um, std(x): {grid_stats['rows']['avg_std_x_um']:.1f} um, std(y): {grid_stats['rows']['avg_std_y_um']:.1f} um",
                    (200, next_line_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            # 列统计
            next_line_y += 30
            cv2.putText(
                display_img,
                f"NCol: {len(columns)}",
                (10, next_line_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )

            if "avg_mean_dy_um" in grid_stats["columns"]:
                cv2.putText(
                    display_img,
                    f"Dy: {grid_stats['columns']['avg_mean_dy_um']:.1f} um, std(y): {grid_stats['columns']['avg_std_y_um']:.1f} um, std(x): {grid_stats['columns']['avg_std_x_um']:.1f} um",
                    (200, next_line_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

    def _draw_row_col_visualization(self, display_img):
        """使用连续色调(hue)绘制行列可视化，并绘制行列平均位置线"""
        if not (self.do_fitting and self.row_col_fitting):
            return

        # 将点分类为行和列
        rows = self.rows
        columns = self.columns

        # 获取ROI和填充信息用于坐标转换
        w_roi, h_roi, ox, oy = self.camera.ROI
        scale = WINDOW_SIZE / max(w_roi, h_roi)
        view_w = int(w_roi * scale)
        view_h = int(h_roi * scale)
        pad_l = (WINDOW_SIZE - view_w) // 2
        pad_t = (WINDOW_SIZE - view_h) // 2

        # 绘制行（水平线）
        for i, row in enumerate(rows):
            if len(row) < 2:
                continue

            # 生成连续色调的颜色 - 对行使用高饱和度
            hue = (i * 30) % 180  # 每行色调差30度，限制在0-180之间避免红色重复
            # 转换HSV到BGR (OpenCV使用BGR)
            color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][
                0
            ].tolist()

            # 计算该行所有点的平均y值
            mean_y = np.mean([s["y"] for s in row])

            # 找出该行中x坐标的最小值和最大值
            min_x = min([s["x"] for s in row])
            max_x = max([s["x"] for s in row])

            # 转换为显示坐标
            s_x1, s_y1 = self.roi_model.coord_in_roi_to_sensor(min_x, mean_y)
            s_x2, s_y2 = self.roi_model.coord_in_roi_to_sensor(max_x, mean_y)

            d_x1, d_y1 = self.roi_model.sensor_to_display(
                s_x1, s_y1, WINDOW_SIZE, pad_l, pad_t
            )
            d_x2, d_y2 = self.roi_model.sensor_to_display(
                s_x2, s_y2, WINDOW_SIZE, pad_l, pad_t
            )

            # 绘制代表该行平均y值的水平实线
            cv2.line(display_img, (d_x1, d_y1), (d_x2, d_y2), color, 2, cv2.LINE_AA)

            # 按x坐标排序
            sorted_row = sorted(row, key=lambda s: s["x"])

            # # 连接每一行的点（用细线）
            # for j in range(len(sorted_row) - 1):
            #     p1 = sorted_row[j]
            #     p2 = sorted_row[j + 1]

            #     # 转换为显示坐标
            #     s_x1, s_y1 = self.roi_model.coord_in_roi_to_sensor(p1["x"], p1["y"])
            #     s_x2, s_y2 = self.roi_model.coord_in_roi_to_sensor(p2["x"], p2["y"])

            #     d_x1, d_y1 = self.roi_model.sensor_to_display(
            #         s_x1, s_y1, WINDOW_SIZE, pad_l, pad_t
            #     )
            #     d_x2, d_y2 = self.roi_model.sensor_to_display(
            #         s_x2, s_y2, WINDOW_SIZE, pad_l, pad_t
            #     )

            #     # 绘制细线连接点
            #     cv2.line(display_img, (d_x1, d_y1), (d_x2, d_y2), color, 1, cv2.LINE_AA)

            # 绘制圆圈标记每个点
            for p in sorted_row:
                s_x, s_y = self.roi_model.coord_in_roi_to_sensor(p["x"], p["y"])
                d_x, d_y = self.roi_model.sensor_to_display(
                    s_x, s_y, WINDOW_SIZE, pad_l, pad_t
                )
                cv2.circle(display_img, (d_x, d_y), 5, color, 2)

        # 绘制列（垂直虚线）
        for i, col in enumerate(columns):
            if len(col) < 2:
                continue

            # 生成连续色调的颜色 - 对列使用中等饱和度和明度
            hue = (i * 30) % 180  # 每列色调差30度
            # 转换HSV到BGR (OpenCV使用BGR)
            color = cv2.cvtColor(np.uint8([[[hue, 180, 255]]]), cv2.COLOR_HSV2BGR)[0][
                0
            ].tolist()

            # 计算该列所有点的平均x值
            mean_x = np.mean([s["x"] for s in col])

            # 找出该列中y坐标的最小值和最大值
            min_y = min([s["y"] for s in col])
            max_y = max([s["y"] for s in col])

            # 转换为显示坐标
            s_x1, s_y1 = self.roi_model.coord_in_roi_to_sensor(mean_x, min_y)
            s_x2, s_y2 = self.roi_model.coord_in_roi_to_sensor(mean_x, max_y)

            d_x1, d_y1 = self.roi_model.sensor_to_display(
                s_x1, s_y1, WINDOW_SIZE, pad_l, pad_t
            )
            d_x2, d_y2 = self.roi_model.sensor_to_display(
                s_x2, s_y2, WINDOW_SIZE, pad_l, pad_t
            )

            # 绘制代表该列平均x值的垂直虚线
            dash_length = 8  # 虚线长度
            gap_length = 4  # 虚线间隔

            # 计算两点间的总距离和方向向量
            dist = np.sqrt((d_x2 - d_x1) ** 2 + (d_y2 - d_y1) ** 2)
            if dist == 0:
                continue

            dx, dy = (d_x2 - d_x1) / dist, (d_y2 - d_y1) / dist

            # 绘制虚线
            pos = 0
            while pos < dist:
                # 虚线段起点
                start_x = int(d_x1 + dx * pos)
                start_y = int(d_y1 + dy * pos)

                # 虚线段终点
                end_pos = min(pos + dash_length, dist)
                end_x = int(d_x1 + dx * end_pos)
                end_y = int(d_y1 + dy * end_pos)

                cv2.line(
                    display_img,
                    (start_x, start_y),
                    (end_x, end_y),
                    color,
                    2,
                    cv2.LINE_AA,
                )

                # 移动到下一段虚线的起点
                pos = end_pos + gap_length

            # 按y坐标排序
            sorted_col = sorted(col, key=lambda s: s["y"])

            # # 连接每一列的点（用细线）
            # for j in range(len(sorted_col) - 1):
            #     p1 = sorted_col[j]
            #     p2 = sorted_col[j + 1]

            #     # 转换为显示坐标
            #     s_x1, s_y1 = self.roi_model.coord_in_roi_to_sensor(p1["x"], p1["y"])
            #     s_x2, s_y2 = self.roi_model.coord_in_roi_to_sensor(p2["x"], p2["y"])

            #     d_x1, d_y1 = self.roi_model.sensor_to_display(
            #         s_x1, s_y1, WINDOW_SIZE, pad_l, pad_t
            #     )
            #     d_x2, d_y2 = self.roi_model.sensor_to_display(
            #         s_x2, s_y2, WINDOW_SIZE, pad_l, pad_t
            #     )

            #     # 绘制细线连接点
            #     cv2.line(display_img, (d_x1, d_y1), (d_x2, d_y2), color, 1, cv2.LINE_AA)

            # 使用方形标记标识列中的每个点
            for p in sorted_col:
                s_x, s_y = self.roi_model.coord_in_roi_to_sensor(p["x"], p["y"])
                d_x, d_y = self.roi_model.sensor_to_display(
                    s_x, s_y, WINDOW_SIZE, pad_l, pad_t
                )
                cv2.drawMarker(display_img, (d_x, d_y), color, cv2.MARKER_SQUARE, 10, 2)

    # ------------------------------------------------------------------ #
    # key handling & helpers
    # ------------------------------------------------------------------ #
    def _handle_key(self, key: int) -> bool:
        if key == 27:  # ESC
            return False
        elif key in (2555904, 2424832, 2490368, 2621440):  # arrows
            self._adjust_exposure(key)
        elif key == ord("a"):
            self.auto_exp = not self.auto_exp
        elif key == ord("f"):
            self.do_fitting = not self.do_fitting
        elif key == ord("g"):
            self.show_stats = not self.show_stats  # toggle stats bar
        elif key == ord("h"):
            self.row_col_fitting = not self.row_col_fitting  # 切换行列拟合模式
        elif key == ord("c"):
            self.rect_sensor = None
        elif key == ord("v"):
            self.fit_rect_sensor = None  # clear fitting rect (#5)
        elif key == ord("s"):
            self._quick_save()
        elif key == ord("d"):
            self._dialog_save()
        elif key == ord("t"):
            self._switch_camera()
        return True

    def _adjust_exposure(self, key):
        if self.auto_exp:
            return
        factor = {2555904: 1.1, 2424832: 0.9, 2490368: 10, 2621440: 0.1}[key]
        self.exposure_us = int(
            np.clip(self.exposure_us * factor, MIN_EXPOSURE, MAX_EXPOSURE)
        )
        self.camera.ExposureTime = self.exposure_us

    def _quick_save(self):
        ts = time.strftime("%Y%m%d_%H%M%S")
        jpg = os.path.join(DATA_DIR, f"vipa_{ts}.jpg")
        npy = os.path.join(DATA_DIR, f"vipa_{ts}.npy")
        frame = self.camera.grab_image()
        cv2.imwrite(jpg, frame)
        np.save(npy, frame)
        logging.info("Saved %s, %s", jpg, npy)

    def _dialog_save(self):
        fp = choose_filename()
        if not fp:
            return
        frame = self.camera.grab_image()
        cv2.imwrite(fp, frame)
        np.save(os.path.splitext(fp)[0] + ".npy", frame)
        logging.info("Saved %s (+ .npy)", fp)

    def _switch_camera(self):
        # save current
        cached_state[self.camera.serial] = dict(
            ROI_model=self.roi_model,
            exp=self.camera.ExposureTime,
            gain=self.camera.Gain,
            auto=self.auto_exp,
            rect=self.rect_sensor,
            fit_rect=self.fit_rect_sensor,
            do_fitting=self.do_fitting,
            show_stats=self.show_stats,
        )
        # next camera
        self.curr_idx = (self.curr_idx + 1) % len(cameras)
        self.camera = cameras[self.curr_idx]
        st = cached_state.get(self.camera.serial)
        if st:
            # load settings
            self.roi_model = st["ROI_model"]
            self.camera.ROI = self.roi_model.tuple
            self.camera.ExposureTime = st["exp"]
            self.camera.Gain = st["gain"]
            self.auto_exp = st["auto"]
            self.rect_sensor = st["rect"]
            self.fit_rect_sensor = st["fit_rect"]
            self.do_fitting = st["do_fitting"]
            self.show_stats = st["show_stats"]
        else:
            # reset settings
            self.roi_model = ROIModel(self.camera.W, self.camera.H)
            self.camera.ROI = self.roi_model.tuple
            self.camera.ExposureTime = self.exposure_us
            self.camera.Gain = 0
            self.auto_exp = False
            self.rect_sensor = None
            self.fit_rect_sensor = None
            self.do_fitting = False
            self.show_stats = False
        # refresh ROI model
        logging.info("Switched to %s", self.camera.model_name)

    def _sleep_for_fps(self, frame_dt):
        target_dt = 1 / FPS_LIMIT
        if frame_dt < target_dt:
            time.sleep(target_dt - frame_dt)
            frame_dt = target_dt
        self.avg_dt = 0.9 * self.avg_dt + 0.1 * frame_dt


# ----------------------------------------------------------------------- #

if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    BaslerViewer().run()
