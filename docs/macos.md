# macOS + FLIR Blackfly S

## This Mac: ready to launch

Double-click **Launch FLIR.command** in the repository folder. It selects FLIR
explicitly: a missing/disconnected camera produces an error, never simulated data.
Double-click **Launch Simulator.command** for synthetic beam arrays.

The installed environment is native Apple Silicon Python 3.12.14 with Spinnaker
4.4.0.246, taken from the SDK you downloaded. The Python environment (`.venv`),
Python runtime (`.python`), and SDK libraries (`.vendor/spinnaker`) are local to
this folder. Homebrew supplies libusb, libomp and FFmpeg 6 dependencies. Keep the
folder in its current location: virtual environments contain absolute paths.
The full SDK installer is excluded. The matching runtime libraries are vendored
for the standalone Mac build and retain their own licenses.

`run.sh` configures the local SDK library and GenTL transport paths automatically.
No global shell-profile changes are required. If you prefer Terminal:

```sh
./run.sh --camera flir
./run.sh --camera flir --mode 8Bit
./run.sh --sim --sim-grid 8x8
./run.sh --camera basler
```

## Downloadable Mac app

GitHub Actions **Build and Release** includes **Build macOS app and DMG
(Apple Silicon)**. Download `pylon_camera-macos-arm64` from the completed run,
or the `.dmg` from its GitHub release. Drag `pylon_camera.app` into Applications.
The app is ad-hoc signed, not Apple-notarized; use macOS Privacy & Security's
Open Anyway option if macOS blocks this trusted download.

The app includes Python, the analysis dependencies, Basler/FLIR support and simulation.
Editable configuration, fitting settings and saved images live under
`~/Library/Application Support/BeamProfiler`, outside the application bundle.
The bundled camera configuration is copied there on first launch without
replacing an existing configuration.

The Apple Silicon DMG bundles the tested Basler and FLIR runtimes, Python, and
all native dependencies. It requires **macOS 26 or later**, but requires no SDK,
Python, Homebrew, or separate camera-library installation. FLIR's libraries are
kept in their own directory to avoid collisions with OpenCV's video libraries.
Third-party libraries retain their original licenses, included inside the app.

The GitHub build verifies the checksum of the copied FLIR runtime, bundles it,
initializes both vendor drivers, and opens a short simulated viewer session.
The source installation steps below are only for running or developing from source.

## Controls and feature parity

The original processing, ROI, statistics, image saving, and HTTP implementations
are shared by FLIR, Basler and simulated cameras.

| Control | Action |
| --- | --- |
| `a` | Automatic exposure, targeting near saturation |
| Left/right arrows | Decrease/increase exposure by 10% |
| Down/up arrows | Divide/multiply exposure by 10 |
| `f` | Beam detection and 2D Gaussian/moment fitting |
| `g` | Beam statistics |
| `h` | Array row/column statistics |
| Wheel | Sensor ROI zoom centered at the pointer |
| Shift + wheel (or Ctrl + wheel) | Change ROI aspect ratio |
| Drag | White measurement rectangle |
| Shift + drag (or Ctrl + drag) | Green fitting rectangle |
| `c` / `v` | Clear white / green rectangle |
| `o` | Live fitting settings, persisted in `fit_config.json` |
| `s` | Save JPEG plus full-depth NumPy frame under `data/` |
| `d` | Choose save location |
| `t` | Switch connected cameras |
| `w` | Existing HTTP interface on port 5000 |
| `y` | Toggle the configured user GPIO output |
| Esc or close window | Quit and release all cameras |

Shift is provided because macOS can intercept Control-click. Arrows support
native Cocoa key codes. The title uses ASCII to avoid OpenCV encoding artifacts;
the native red close button is enabled and requests a clean camera shutdown. Tk initializes before OpenCV and its events run on the
main thread, avoiding both background-thread and NSApplication startup crashes.

The HTTP server remains the upstream interface (see `web_api.md`), including
frame/spot/rectangle readout and fitting-parameter editing. It does not add remote
exposure/gain commands. The upstream server binds all interfaces on port 5000.
CUDA acceleration is unavailable on Apple Silicon; the same fitting calculations
run on the CPU. No Metal acceleration is claimed.

## Your BFS-U3-31S4M-C

- Detected model string: `Blackfly S BFS-U3-31S4M`.
- Full sensor: 2048 × 1536 pixels; pixel pitch: 3.45 µm.
- Native Mono8 and Mono16 modes; the 12-bit ADC is MSB-aligned in the 16-bit
  container. Frames are copied before SDK buffers are released, preserving raw
  values for fitting and `.npy` saving.
- Exposure/gain writes clamp to the camera's reported limits. ROI values align
  to hardware increments; acquisition pauses and restarts during ROI changes.
- Software FrameStart triggering is used when supported. If configuration fails,
  TriggerMode is explicitly disabled before free-running acquisition.
- Unknown FLIR models require their real pixel pitch in `camera_config.yaml`;
  the driver will not invent a calibration.

The pixel pitch describes the sensor plane. Optical magnification is not
calibrated by this change.

### Hardware sync

The model configuration maps inverted ExposureActive to **Line1 (pin 4, white)**,
and the `y`-controlled UserOutput0 to **Line2 (pin 3, red)**. This differs from the
original Basler Line2/Line3 mapping. Line3 on this Blackfly is an input/power pin.
No 3.3 V auxiliary power output is enabled by this application. Use the camera's
GPIO electrical specifications for pull-ups, grounding, and external connections.
Remove the `sync` entries to leave those features unconfigured.

ExposureActive configuration was accepted by the connected camera. Buffer-release
and user-pulse restoration on acquisition failure are covered by automated tests.
Electrical polarity, timing, and external equipment synchronization have **not**
been measured with a scope. The software user pulse spans acquisition/transfer;
ExposureActive is the output corresponding to the actual sensor exposure window.

## Recreate on another Mac

Use Python **3.12**, a matching Mac architecture, and matching Spinnaker runtime
and PySpin versions. The downloaded 4.4.0.246 installer includes the cp312 ARM64
Python package, so a separate Python download was unnecessary on this Mac.

1. Install the prerequisites listed by the SDK:
   `brew install pkg-config libomp libusb ffmpeg@6`.
2. Download/install Spinnaker for your Mac from Teledyne. Follow its GenTL setup
   instructions; `SPINNAKER_GENTL64_CTI` must point to `Spinnaker_GenTL.cti`.
3. Create an environment and install the base dependencies:

   ```sh
   python3.12 -m venv .venv
   .venv/bin/python -m pip install -r requirements.txt
   ```

4. Unpack the matching Python archive from `/Applications/Spinnaker/PySpin` and
   install its `spinnaker_python-...cp312...arm64.whl` with the environment's pip.
   Do **not** install the unrelated `pyspin` package from PyPI.
5. Run `./run.sh --camera flir`. Use the vendor viewer to diagnose USB access if
   discovery fails, and close it before opening the beam profiler.

For Basler, additionally install `requirements-basler.txt`. Pylon is optional
for FLIR-only use and simulation.

## Verification

Final test suite: **40 passed, 5 skipped** (CUDA-only tests skipped on this Mac).

On this Mac with the connected Blackfly:

- Full-resolution Mono8 and Mono16 acquisition, software triggering.
- ROI request `(511,383,101,99)` aligned to `(508,382,100,98)`, followed by
  successful restoration of full-frame acquisition.
- Exposure, gain, and application auto-exposure writes/readback.
- OpenCV viewer with fitting/statistics enabled; settings opened, hidden, reopened.
- Flask test-client responses for status, images, spots, settings and rectangles.
- Saved JPEG and exact raw NumPy round-trip.

The captured scene was dark; live beam metrology was not validated against a
physical reference. Synthetic known-truth tests validate fitting and grid analysis.
Actual live evidence is in `data/validation/hardware.json` and `viewer.jpg`.
`tools/check_hardware.py` reproduces the native GUI/hardware acceptance test;
run it with the same SDK environment variables as `run.sh`. It changes exposure,
gain and ROI, so close the regular viewer first.

## Provenance and references

Built on tim4431/Basler_Beam_Profiler commit
`a21b706350269a2db56d0125f2a4fe5472789025`. Compared laserroger's earlier monolithic
port at `77aabeb39025d9056fa43d2c4fca847ed122abad`; retained the current upstream's
newer modular processing and fitting settings rather than replacing it with the
older program.

- [Original project](https://github.com/tim4431/Basler_Beam_Profiler)
- [Earlier port](https://github.com/laserroger/Basler_Beam_Profiler)
- [Spinnaker download](https://www.teledynevisionsolutions.com/products/spinnaker-sdk/)
- [Python compatibility](https://softwareservices.flir.com/Spinnaker/latest/getting-started/python.html)
- [Camera specifications](https://softwareservices.flir.com/BFS-U3-31S4/latest/Model/spec.html)
- [GPIO definitions](https://softwareservices.flir.com/BFS-U3-31S4/latest/Model/public/DigitalIOControl.html)
