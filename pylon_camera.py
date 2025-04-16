import cv2, sys, time, logging, math, os, tkinter as tk
import numpy as np  # Needed for image processing and math
from tkinter import filedialog  # Needed for file dialog

# Configure logging to stdout with timestamp and level
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

from basler import *  # or use a different subclass if appropriate
from blob_detector import blob_detector

MIN_EXPOSURE = 30
MAX_EXPOSURE = 100000
FPS_LIMIT = 60
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# --- Global parameters ---
roi_scale = 1.0  # Fraction of the full sensor view (1.0 = full sensor)
auto_exposure_enabled = True
do_fitting = False  # Toggle for displaying beam spots
stats_enabled = False  # Toggle for showing spot statistics

# Variables for saving hint message (displayed for 1 second)
save_hint = ""
save_hint_end_time = 0

# Create camera instance in 8-bit mode.
camera = Baslera2A4504(mode="16Bit")
camera.set_default_roi()  # Set ROI to full sensor
exposure_time = camera.ExposureTime
W = camera.W
H = camera.H
D = 1200
display_width = D
display_height = int(D * H / W)

# Global variable to store last mouse sensor coordinate (in pixels)
last_mouse_sensor = (0, 0)

# Variables for rectangle drawing.
rect_start_disp = None  # Starting display coordinate of rectangle (x,y)
rect_end_disp = None  # Ending display coordinate of rectangle (x,y)
drawing = False  # True while left mouse button is held down.
rect_sensor_coords = None  # Finalized rectangle in sensor coordinates (x1, y1, x2, y2)


def update_roi(new_scale, mouse_x, mouse_y):
    global roi_scale
    new_scale = max(min(new_scale, 1), 0.1)
    if abs(new_scale - roi_scale) < 1e-3:
        return  # no significant change
    roi_scale = new_scale

    new_width = int(W * roi_scale)
    new_height = int(H * roi_scale)

    curr_roi = camera.ROI
    curr_width, curr_height, curr_offset_x, curr_offset_y = curr_roi

    r_x = mouse_x / display_width
    r_y = mouse_y / display_height

    m_x = curr_offset_x + r_x * curr_width
    m_y = curr_offset_y + r_y * curr_height

    new_offset_x = int(m_x - r_x * new_width)
    new_offset_y = int(m_y - r_y * new_height)

    new_offset_x = max(0, min(new_offset_x, W - new_width))
    new_offset_y = max(0, min(new_offset_y, H - new_height))

    camera.ROI = (new_width, new_height, new_offset_x, new_offset_y)


def mouse_callback(event, x, y, flags, param):
    global roi_scale, last_mouse_sensor
    global rect_start_disp, rect_end_disp, drawing, rect_sensor_coords

    if event == cv2.EVENT_MOUSEWHEEL:
        if flags > 0:  # wheel up -> zoom in
            new_scale = roi_scale - 0.05
        else:  # wheel down -> zoom out
            new_scale = roi_scale + 0.05
        update_roi(new_scale, x, y)

    if event == cv2.EVENT_MOUSEMOVE:
        curr_roi = camera.ROI
        roi_width, roi_height, offset_x, offset_y = curr_roi
        r_x = x / display_width
        r_y = y / display_height
        sensor_x = offset_x + r_x * roi_width
        sensor_y = offset_y + r_y * roi_height
        last_mouse_sensor = (int(sensor_x), int(sensor_y))

        if drawing:
            rect_end_disp = (x, y)

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        rect_start_disp = (x, y)
        rect_end_disp = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        rect_end_disp = (x, y)
        roi_width, roi_height, offset_x, offset_y = camera.ROI
        sensor_x1 = offset_x + (rect_start_disp[0] / display_width) * roi_width
        sensor_y1 = offset_y + (rect_start_disp[1] / display_height) * roi_height
        sensor_x2 = offset_x + (rect_end_disp[0] / display_width) * roi_width
        sensor_y2 = offset_y + (rect_end_disp[1] / display_height) * roi_height
        rect_sensor_coords = (
            int(min(sensor_x1, sensor_x2)),
            int(min(sensor_y1, sensor_y2)),
            int(max(sensor_x1, sensor_x2)),
            int(max(sensor_y1, sensor_y2)),
        )


cv2.namedWindow("Basler Camera", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Basler Camera", 1000, 1000)
cv2.setMouseCallback("Basler Camera", mouse_callback)

avg_dt = 1 / FPS_LIMIT


def get_filename_tk():
    root = tk.Tk()
    root.withdraw()
    # Open a file dialog to select the save location and filename.
    # default filename is DATA_DIR/vipa_YYYYMMDD_HHMMSS.jpg
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"vipa_{timestamp}.jpg"
    file_path = filedialog.asksaveasfilename(
        initialdir=DATA_DIR,
        initialfile=filename,
        title="Save Image",
        filetypes=(("JPEG files", "*.jpg"), ("All files", "*.*")),
    )
    root.destroy()
    return file_path


while True:
    loop_start = time.time()

    # Grab an image (grayscale).
    img = camera.grab_image()
    # convert to 8Bit if the image is in 16Bit mode
    if img is None:
        continue
    if camera.mode == "16Bit":
        img_disp = cv2.convertScaleAbs(img, alpha=1.0 / 256.0)
    else:
        img_disp = img

    # Convert the grayscale image to RGB for display.
    # (This does not affect blob detection, which works on the original 'img'.)
    img_rgb = cv2.cvtColor(img_disp, cv2.COLOR_GRAY2RGB)

    # Variables to hold stats values (if computed)
    stats_std_x = None
    stats_std_y = None

    # If overlaying blob detection results, work on the grayscale image for detection
    # and then draw colored overlays on the RGB image.
    if do_fitting:
        spots = blob_detector(img)
        # Process each detected spot.
        for spot in spots:
            x = int(round(spot["x"]))
            y = int(round(spot["y"]))
            sigma0 = spot["sigma_0"]
            sigma1 = spot["sigma_1"]
            vec0 = spot["vec_0"]
            angle = np.degrees(np.arctan2(vec0[1], vec0[0]))
            axes = (int(round(sigma0)), int(round(sigma1)))
            cv2.ellipse(img_rgb, (x, y), axes, angle, 0, 360, (0, 255, 0), 2)
            angle_rad = np.radians(angle)
            x_major = int(round(x + sigma0 * np.cos(angle_rad)))
            y_major = int(round(y + sigma0 * np.sin(angle_rad)))
            x_minor = int(round(x + sigma1 * np.cos(angle_rad + math.pi / 2)))
            y_minor = int(round(y + sigma1 * np.sin(angle_rad + math.pi / 2)))
            cv2.line(img_rgb, (x, y), (x_major, y_major), (255, 0, 0), 2)
            cv2.line(img_rgb, (x, y), (x_minor, y_minor), (0, 0, 255), 2)
            cv2.putText(
                img_rgb,
                "0",
                (x_major, y_major),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                1,
            )
            cv2.putText(
                img_rgb,
                "1",
                (x_minor, y_minor),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                1,
            )
            _Y_spacing = 20
            _Y_start = int(np.sqrt(sigma0 * sigma1) + 10 + _Y_spacing)
            cv2.putText(
                img_rgb,
                f"({x}, {y})",
                (x, y + _Y_start),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                1,
            )
            sigma0_um = sigma0 * camera.pixel_size * 1e6
            sigma1_um = sigma1 * camera.pixel_size * 1e6
            cv2.putText(
                img_rgb,
                f"s0={sigma0_um:.1f} um",
                (x, y + _Y_start + _Y_spacing),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                1,
            )
            cv2.putText(
                img_rgb,
                f"s1={sigma1_um:.1f} um",
                (x, y + _Y_start + 2 * _Y_spacing),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                1,
            )
        # If stats are toggled, compute the statistics from the spots.
        if stats_enabled:
            if len(spots) >= 2:
                # Sort spots by x-coordinate.
                sorted_spots = sorted(spots, key=lambda spot: spot["x"])
                xs = [int(round(spot["x"])) for spot in sorted_spots]
                ys = [int(round(spot["y"])) for spot in sorted_spots]
                dxs = [xs[i + 1] - xs[i] for i in range(len(xs) - 1)]
                dys = [ys[i + 1] - ys[i] for i in range(len(ys) - 1)]
                stats_std_x = np.std(dxs)
                stats_std_y = np.std(dys)
                stats_mean_dx = np.mean(dxs)
                stats_mean_dy = np.mean(dys)
                stats_mean_dx_um = stats_mean_dx * camera.pixel_size * 1e6
                stats_mean_dy_um = stats_mean_dy * camera.pixel_size * 1e6
            else:
                stats_std_x = 0
                stats_std_y = 0
                stats_mean_dx = 0
                stats_mean_dy = 0

    # Resize the RGB image for display.
    display_img = cv2.resize(
        img_rgb, (display_width, display_height), interpolation=cv2.INTER_LINEAR
    )

    # Retrieve current ROI info.
    curr_roi = camera.ROI  # (roi_width, roi_height, offset_x, offset_y)
    curr_width, curr_height, curr_offset_x, curr_offset_y = curr_roi

    roi_info = f"ROI: Offset=({curr_offset_x},{curr_offset_y}) Size=({curr_width}x{curr_height})"
    if auto_exposure_enabled:
        exp_info = f"Exposure = Auto, {camera.ExposureTime} us"
    else:
        exp_info = f"Exposure = {exposure_time} us"
    fps_info = f"FPS: {1/avg_dt:.1f}"
    mouse_info = f"Mouse: ({last_mouse_sensor[0]}, {last_mouse_sensor[1]}) px"

    # Updated hint information to include the new save keybindings.
    hint_info = (
        "Arrow keys adjust exposure; 'a' toggles auto; 'c' clears rect; "
        "'f' toggles spots; 'g' toggles stats; 's' quick save; 'd' save as...; ESC to exit."
    )

    line_height = 30
    start_y = 30
    cv2.putText(
        display_img,
        roi_info + " " + mouse_info,
        (10, start_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        display_img,
        exp_info,
        (10, start_y + line_height),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        display_img,
        fps_info,
        (10, start_y + 2 * line_height),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        display_img,
        hint_info,
        (10, display_height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Display save hint if active.
    if time.time() < save_hint_end_time:
        cv2.putText(
            display_img,
            save_hint,
            (10, start_y + 4 * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    # If stats were computed, overlay their info and draw a bar indicator.
    if stats_enabled and stats_std_x is not None:
        bar_factor = 5  # scaling factor (adjust as needed)
        max_bar_length = 100  # maximum bar length for display
        bar_y = start_y + 3 * line_height
        text_x = 10

        cv2.putText(
            display_img,
            f"std_x: {stats_std_x:.2f}",
            (text_x, bar_y + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )
        bar_x = text_x + 150
        bar_length = min(int(stats_std_x * bar_factor), max_bar_length)
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
            f"std_y: {stats_std_y:.2f}",
            (text_x, bar_y + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )
        bar_x = text_x + 150
        bar_length = min(int(stats_std_y * bar_factor), max_bar_length)
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
        # also show the mean dx and dy values, in um
        text_mean_dx_x = bar_x + max_bar_length + 50
        cv2.putText(
            display_img,
            f"Dx: {stats_mean_dx_um:.1f} um",
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
            f"Dy: {stats_mean_dy_um:.1f} um",
            (text_mean_dy_x, bar_y + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )

    # Draw rectangle if in drawing mode or if one has been finalized.
    if drawing and rect_start_disp is not None and rect_end_disp is not None:
        x1, y1 = rect_start_disp
        x2, y2 = rect_end_disp
        tl = (min(x1, x2), min(y1, y2))
        br = (max(x1, x2), max(y1, y2))
        cv2.rectangle(display_img, tl, br, (255, 255, 255), 2)
    elif rect_sensor_coords is not None:
        roi_width, roi_height, offset_x, offset_y = camera.ROI
        disp_x1 = int((rect_sensor_coords[0] - offset_x) / roi_width * display_width)
        disp_y1 = int((rect_sensor_coords[1] - offset_y) / roi_height * display_height)
        disp_x2 = int((rect_sensor_coords[2] - offset_x) / roi_width * display_width)
        disp_y2 = int((rect_sensor_coords[3] - offset_y) / roi_height * display_height)
        cv2.rectangle(
            display_img, (disp_x1, disp_y1), (disp_x2, disp_y2), (255, 255, 255), 2
        )
        rect_width_px = rect_sensor_coords[2] - rect_sensor_coords[0]
        rect_height_px = rect_sensor_coords[3] - rect_sensor_coords[1]
        rect_width_um = rect_width_px * camera.pixel_size * 1e6
        rect_height_um = rect_height_px * camera.pixel_size * 1e6
        rect_info = f"Rect: {rect_width_px}px x {rect_height_px}px ({rect_width_um:.1f}um x {rect_height_um:.1f}um)"
        cv2.putText(
            display_img,
            rect_info,
            (10, start_y + 3 * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.imshow("Basler Camera", display_img)

    # key = cv2.waitKey(1) & 0xFF
    key = cv2.waitKeyEx(1)
    # print(key)
    if key == 27:  # ESC key
        break
    elif (
        key == 2555904 and not auto_exposure_enabled
    ):  # Right arrow: increase exposure additively (+20)
        exposure_time = min(MAX_EXPOSURE, exposure_time + 20)
        camera.ExposureTime = int(exposure_time)
    elif (
        key == 2424832 and not auto_exposure_enabled
    ):  # Left arrow: decrease exposure additively (-20)
        exposure_time = max(MIN_EXPOSURE, exposure_time - 20)
        camera.ExposureTime = int(exposure_time)
    elif (
        key == 2490368 and not auto_exposure_enabled
    ):  # Up arrow: multiplicatively increase exposure (*10)
        exposure_time = min(MAX_EXPOSURE, exposure_time * 10)
        camera.ExposureTime = int(exposure_time)
    elif (
        key == 2621440 and not auto_exposure_enabled
    ):  # Down arrow: multiplicatively decrease exposure (/10)
        exposure_time = max(MIN_EXPOSURE, exposure_time / 10)
        camera.ExposureTime = int(exposure_time)
    elif key == ord("f"):
        do_fitting = not do_fitting
    elif key == ord("g"):
        stats_enabled = not stats_enabled
    elif key == ord("a"):
        auto_exposure_enabled = not auto_exposure_enabled
        if auto_exposure_enabled:
            try:
                camera.AutoExposureOn(range=(MIN_EXPOSURE, MAX_EXPOSURE))
                logging.info("Auto Exposure Enabled")
            except Exception as e:
                logging.info("Failed to enable auto exposure:", e)
        else:
            exposure_time = camera.ExposureTime
            camera.AutoExposureOff()
            logging.info("Auto Exposure Disabled")
    elif key == ord("c"):
        rect_sensor_coords = None
        rect_start_disp = None
        rect_end_disp = None
    elif key == ord("s"):
        # Quick save using a timestamp-based filename.
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename_jpg = f"vipa_{timestamp}.jpg"
        filename_npy = f"vipa_{timestamp}.npy"
        cv2.imwrite(os.path.join(DATA_DIR, filename_jpg), img_disp)
        np.save(os.path.join(DATA_DIR, filename_npy), img)
        save_hint = f"Saved {filename_jpg} & {filename_npy}"
        save_hint_end_time = time.time() + 2
    elif key == ord("d"):
        file_path = get_filename_tk()
        if not file_path:
            logging.info("Save cancelled.")
        cv2.imwrite(file_path, img_disp)
        base = os.path.splitext(file_path)[0]
        npy_path = base + ".npy"
        np.save(npy_path, img)
        save_hint = f"Saved {file_path} & {npy_path}"
        save_hint_end_time = time.time() + 2

    loop_end = time.time()
    frame_dt = loop_end - loop_start
    avg_dt = 0.9 * avg_dt + 0.1 * frame_dt

    target_dt = 1 / FPS_LIMIT
    if frame_dt < target_dt:
        time.sleep(target_dt - frame_dt)
        avg_dt = 0.9 * avg_dt + 0.1 * target_dt

camera.close()
cv2.destroyAllWindows()
