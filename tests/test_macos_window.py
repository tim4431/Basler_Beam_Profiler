"""Exercise Cocoa/Tk initialization and the red close button in a fresh process."""
import subprocess
import sys

import pytest


@pytest.mark.skipif(sys.platform != 'darwin', reason='Cocoa is macOS-only')
def test_native_close_button_and_title():
    script = '''
import tkinter as tk
from beam_profiler.ui import settings
try:
    settings.prepare_gui()
except tk.TclError:
    raise SystemExit(77)
from beam_profiler.cameras import SimulatedCamera
from beam_profiler.ui.viewer import Viewer, WINDOW
from AppKit import NSApplication, NSWindowCloseButton
import cv2
camera = SimulatedCamera()
viewer = Viewer([camera])
try:
    cv2.imshow(WINDOW, cv2.convertScaleAbs(camera.grab_image(), alpha=1/256))
    cv2.waitKeyEx(10)
    settings.open_settings()
    settings.pump_events()
    settings.close_settings()
    settings.pump_events()
    settings.open_settings()
    settings.pump_events()
    window = next(w for w in NSApplication.sharedApplication().windows()
                  if str(w.title()) == WINDOW)
    assert str(window.title()) == 'Beam Profiler - Basler / FLIR'
    button = window.standardWindowButton_(NSWindowCloseButton)
    assert button.isEnabled()
    button.performClick_(None)
    assert viewer._quit_requested
    viewer.run()
finally:
    camera.close()
    cv2.destroyAllWindows()
    settings.destroy_settings()
'''
    result = subprocess.run([sys.executable, '-c', script], capture_output=True,
                            text=True, timeout=30)
    if result.returncode == 77:
        pytest.skip('No native desktop session available')
    assert result.returncode == 0, result.stdout + result.stderr
