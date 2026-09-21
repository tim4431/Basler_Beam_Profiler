"""Check that all non-system FLIR links resolve inside the packaged application."""
import argparse
from pathlib import Path
from tools.bundle_flir import dependencies


def check(app):
    frameworks = app / 'Contents/Frameworks'
    binaries = list((frameworks / 'flir').iterdir()) + list(frameworks.glob('_PySpin*.so'))
    if not binaries:
        raise RuntimeError('No bundled FLIR runtime found')
    for binary in binaries:
        for dep in dependencies(binary):
            if dep.startswith(('/usr/lib/', '/System/Library/')) or dep == binary.name:
                continue  # system dependencies or the CTI's own install ID
            if dep.startswith('@loader_path/'):
                target = binary.parent / dep.removeprefix('@loader_path/')
            elif dep.startswith('@rpath/flir/'):
                target = frameworks / dep.removeprefix('@rpath/')
            else:
                raise RuntimeError(f'External dependency: {binary}: {dep}')
            if not target.is_file():
                raise RuntimeError(f'Missing dependency: {binary}: {dep}')
    print(f'Checked {len(binaries)} FLIR binaries: all non-system dependencies are inside the app')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('app', type=Path)
    check(parser.parse_args().app)
