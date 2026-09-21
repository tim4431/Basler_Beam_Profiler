"""Copy a matching FLIR runtime into a Mac app, isolated from OpenCV's libs.

The runtime input contains lib/, python/PySpin.py, python/_PySpin*.so and licenses/.
All non-system native dependencies must exist in lib/ or at their linked paths.
"""
import argparse
import os
from pathlib import Path
import shutil
import subprocess


def dependencies(path):
    output = subprocess.check_output(['otool', '-arch', 'arm64', '-L', str(path)], text=True)
    return [line.strip().split(' (compatibility')[0] for line in output.splitlines() if ' (compatibility' in line]


def bundle(app, runtime):
    frameworks = app / 'Contents/Frameworks'
    target = frameworks / 'flir'
    target.mkdir(parents=True, exist_ok=True)
    lib = runtime / 'lib'
    copied = {}

    def resolve(name, parent):
        for candidate in (lib / Path(name).name,
                          parent / name.replace('@loader_path/', ''),
                          parent / Path(name).name, Path(name),
                          Path('/opt/homebrew/lib') / Path(name).name):
            if candidate.is_file():
                return candidate.resolve()
        raise RuntimeError(f'Missing FLIR runtime dependency: {name}')

    def copy_library(source, name=None):
        name = name or source.name
        if name in copied:
            return
        destination = target / name
        copied[name] = source
        shutil.copyfile(source, destination)
        os.chmod(destination, 0o755)
        for dep in dependencies(source):
            if dep.startswith(('/usr/lib/', '/System/Library/')):
                continue
            dep_source = resolve(dep, source.parent)
            dep_name = Path(dep).name
            # The first otool entry of a dylib is its own install ID.
            if dep_source == source.resolve():
                continue
            copy_library(dep_source, dep_name)
            subprocess.run(['install_name_tool', '-change', dep,
                            '@loader_path/' + dep_name, str(destination)], check=True,
                           capture_output=True)
        if destination.suffix == '.dylib':
            subprocess.run(['install_name_tool', '-id', '@rpath/flir/' + name,
                            str(destination)], check=True, capture_output=True)
        subprocess.run(['codesign', '--force', '--sign', '-', str(destination)],
                       check=True, capture_output=True)

    binding = next((runtime / 'python').glob('_PySpin*.so'))
    for dep in dependencies(binding):
        if not dep.startswith(('/usr/lib/', '/System/Library/')):
            copy_library(resolve(dep, binding.parent), Path(dep).name)
    transport = lib / 'spinnaker-gentl/Spinnaker_GenTL.cti'
    copy_library(transport)
    destination = frameworks / binding.name
    shutil.copyfile(binding, destination)
    os.chmod(destination, 0o755)
    for dep in dependencies(binding):
        if not dep.startswith(('/usr/lib/', '/System/Library/')):
            subprocess.run(['install_name_tool', '-change', dep,
                            '@loader_path/flir/' + Path(dep).name, str(destination)],
                           check=True, capture_output=True)
    shutil.copyfile(runtime / 'python/PySpin.py', frameworks / 'PySpin.py')
    shutil.copytree(runtime / 'licenses', app / 'Contents/Resources/licenses/spinnaker',
                    dirs_exist_ok=True)
    subprocess.run(['codesign', '--force', '--sign', '-', str(destination)], check=True,
                   capture_output=True)
    subprocess.run(['codesign', '--force', '--deep', '--sign', '-', str(app)], check=True,
                   capture_output=True)
    print(f'Bundled PySpin and {len(copied)} matching native libraries in {app}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--app', type=Path, required=True)
    parser.add_argument('--runtime', type=Path, default=Path('.vendor/spinnaker'))
    args = parser.parse_args()
    bundle(args.app, args.runtime)
