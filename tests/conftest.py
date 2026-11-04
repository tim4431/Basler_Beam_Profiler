"""Keep the test suite away from the user's real config files.

`fitconfig` loads fit_config.json at import, so without this every test would
run against whatever the last GUI session left behind - and would overwrite it.
`cameraconfig` is cached on first use, so an edited camera_config.yaml would
otherwise change what the simulated camera reports.
"""

import pytest

from beam_profiler import cameraconfig, fitconfig


@pytest.fixture(autouse=True)
def isolated_fit_config(tmp_path_factory, monkeypatch):
    """Every test starts from the defaults and writes only to a temp file."""
    path = tmp_path_factory.mktemp("fitcfg") / "fit_config.json"
    monkeypatch.setattr(fitconfig, "CONFIG_PATH", str(path))
    fitconfig.set_active(fitconfig.FitConfig(), save=False)
    yield
    fitconfig.set_active(fitconfig.FitConfig(), save=False)


@pytest.fixture(autouse=True)
def isolated_camera_config(monkeypatch):
    """No camera_config.yaml, so every camera resolves to the built-in defaults.
    A test that wants a file calls `cameraconfig.reload(path)` itself."""
    monkeypatch.setattr(cameraconfig, "_cached", cameraconfig.CameraConfigFile())
    yield
    monkeypatch.setattr(cameraconfig, "_cached", None)
