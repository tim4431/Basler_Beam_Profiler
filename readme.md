# Basler Beam Profiler

The Basler Beam Profiler is a Python application designed to interface with Basler cameras for beam profiling tasks.

![Demo](docs/demo.png)


## Usage
### Running the Application
```bash
pip install -r requirements.txt
python beam_profiler.py
```


### Configuring the camera

`camera_config.yaml`

```yaml
camera:
  a2A5060-15umBAS:                     # the name should match the camera name shown in pylon_camera
    default_roi: [5060, 5060, 4, 4]    # [width, height, x_pad, y_pad]
    pixel_size: 2.5e-6                 # pixel size in meters, i.e. 2.5um
```
### Using the compiled application
First clone this repo, setup `camera_config.yaml`

Check release to download the latest version: [Releases](https://github.com/tim4431/Basler_Beam_Profiler/releases)

Put the `pylon_camera.exe` executable in the same directory as `camera_config.yaml`.

### Keyboard Shortcuts
- `Esc`: Quit the application
- `arrow keys`: up/down=change exposure by 10 times, left/right=change exposure by 10%
- `a`: toggle auto-exposure (auto exposure adjusts the exposure time to keep the max intensity to be near-saturated)
- `f`: toggle blob fitting
- `g`: toggle statistics showing (statistics of fitted blobs)
- `h`: toggle row/col statistics (works only for beam arrays)
- `mouse drag`: create a white rectangle in the canvas
- `c`: clear the white rectangle
- `ctrl + mouse drag`: create a green rectangle in the canvas, only blobs inside the rectangle will be fitted
- `v`: clear the green rectangle
- `s`: quick save the current frame
- `d`: save the current frame with dialogue box, allowing the user to choose the file location and name
- `t`: switch to the next camera (if multiple cameras are connected)
- `mouse wheel`: zoom in/out the canvas