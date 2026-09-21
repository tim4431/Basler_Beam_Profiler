"""Optional live-camera acceptance test: python -m tools.check_hardware.

Temporarily changes exposure, gain and ROI; saves frames under data/validation.
Exercises native GUI + settings and then exits. Does not toggle user GPIO.
"""
import json
from pathlib import Path
import cv2
import numpy as np
from beam_profiler.cameras import open_cameras
from beam_profiler.ui.viewer import Viewer
from beam_profiler.ui import settings
from beam_profiler.server import create_app


def main():
    results = {}
    out = Path('data/validation')
    out.mkdir(parents=True, exist_ok=True)
    for mode in ('8Bit', '16Bit'):
        cameras = open_cameras(backend='flir', mode=mode)
        c = cameras[0]
        try:
            c.ExposureTime = 1000
            f = c.grab_image()
            assert f.shape == (c.H, c.W)
            c.ROI = (511, 383, 101, 99)
            cropped = c.grab_image()
            roi = c.ROI
            assert cropped.shape == (roi[1], roi[0])
            c.ROI = (c.W, c.H, 0, 0)
            assert c.grab_image().shape == (c.H, c.W)
            c.AutoExposureOn(range=(30, 100000))
            for _ in range(3): c.grab_image()
            c.AutoExposureOff()
            c.Gain = 1
            assert abs(c.Gain - 1) < 0.2
            c.Gain = 0
            results[mode] = dict(shape=list(f.shape), dtype=str(f.dtype), min=int(f.min()), max=int(f.max()),
                                 cropped_roi=list(roi), auto_exposure_us=c.ExposureTime,
                                 software_trigger=c.software_trigger, exposure_sync=c.exposure_line_available)
            print(mode, results[mode], flush=True)
        finally:
            for camera in cameras: camera.close()
    cameras = open_cameras(backend='flir')
    try:
        v = Viewer(cameras)
        v.do_fitting = v.show_stats = v.row_col_fitting = True
        count = 0
        original = v._handle_key
        def step(key):
            nonlocal count
            count += 1
            if count == 2: original(ord('o'))
            if count == 5: settings.close_settings()
            if count == 7: original(ord('o'))
            if count == 10:
                client = create_app(v).test_client()
                for endpoint in ('status', 'image', 'spots', 'fit_config'):
                    assert client.get('/api/' + endpoint).status_code == 200, endpoint
                assert client.post('/api/set_rect', json={'coords':[10, 10, 200, 200]}).status_code == 200
                assert client.get('/api/rect_stats').status_code == 200
                v._save_frame(str(out / 'flir-live.jpg'))
                assert np.array_equal(np.load(out / 'flir-live.npy'), v.current_frame)
                cv2.imwrite(str(out / 'viewer.jpg'), v._compose(v.frame_disp, v.latest_spots))
                results['gui'] = dict(frames=count, settings_reopened=True, api=True, raw_save_roundtrip=True)
                return False
            return original(key)
        v._handle_key = step
        v.run()
    finally:
        for c in cameras: c.close()
    (out / 'hardware.json').write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == '__main__': main()
