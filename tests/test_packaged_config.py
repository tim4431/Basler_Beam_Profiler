from pathlib import Path
from beam_profiler import config


def test_mac_bundle_uses_writable_config_without_overwriting(tmp_path, monkeypatch):
    bundle = tmp_path / 'bundle'
    bundle.mkdir()
    (bundle / 'camera_config.yaml').write_text('cameras: {}\n')
    writable = tmp_path / 'support'
    monkeypatch.setattr(config.sys, 'frozen', True, raising=False)
    monkeypatch.setattr(config.sys, 'platform', 'darwin')
    monkeypatch.setattr(config.sys, '_MEIPASS', str(bundle), raising=False)
    monkeypatch.setattr(config.os.path, 'expanduser', lambda _: str(writable))
    assert config.app_dir() == str(writable)
    path = writable / 'camera_config.yaml'
    assert path.read_text() == 'cameras: {}\n'
    path.write_text('cameras: {custom: 1}\n')
    config.app_dir()
    assert path.read_text() == 'cameras: {custom: 1}\n'


def test_windows_bundle_keeps_executable_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(config.sys, 'frozen', True, raising=False)
    monkeypatch.setattr(config.sys, 'platform', 'win32')
    monkeypatch.setattr(config.sys, 'executable', str(tmp_path / 'profiler.exe'))
    assert config.app_dir() == str(tmp_path)
