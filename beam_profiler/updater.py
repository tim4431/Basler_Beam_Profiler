"""Keep the app up to date with the GitHub repository.

Two kinds of install update themselves:

* **A git checkout** (run.sh, the desktop icon, ``python -m beam_profiler``).
  A background thread fetches the current branch's upstream; at the next start,
  before anything else is imported, ``main()`` fast-forwards to it, reinstalls
  requirements.txt if that changed, and re-executes itself into the new code.
  Only ever a fast-forward of a clean tree on a branch: local changes to
  tracked files, local commits the upstream lacks, or a detached HEAD leave the
  checkout alone (logged, not toasted - that is a developer's machine).  If
  pip fails the merge is undone, so a missing dependency cannot stop the app
  from starting; the next start tries again.
* **The Windows exe** built by CI.  CI stamps the release tag into
  ``_build.py``; the latest GitHub release is compared with it, and a newer
  ``pylon_camera.exe`` is downloaded to ``pylon_camera.exe.new``, checked
  against the size and SHA-256 GitHub reports.  At the next start (or U) it
  is renamed into place - a running exe can be renamed, not overwritten - and
  started; the old one is deleted the time after.

Nothing changes under a running app, this one or another copy: a one-file exe
reads every lazy import from its own file *by path*, and a checkout imports
`server` and the Tk windows on demand, so new files under a running process
would break it later.  The swap and the merge therefore happen only at start-up,
before the package is imported, and only when no other copy of the app from
this folder is running (each holds a lock on ``data/.instances/<pid>``).
No network, no git, a read-only folder - every failure is a log line and an
``"error"`` status, never an exception.  ``--no-update``,
``BEAM_PROFILER_UPDATE=0`` or ``"auto_update": false`` in ui_config.json turn
it off.

Only the standard library and `config`/`uiconfig` are imported: `main()` runs
`apply_at_startup` before the rest of the package is loaded.
"""

from __future__ import annotations

import glob
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
import urllib.request
from dataclasses import asdict, dataclass

from .config import APP_DIR

try:  # written by CI into release builds only (.github/workflows/build.yml)
    from ._build import RELEASE  # e.g. "v2026.11.05.0048"
except ImportError:
    RELEASE = None

GITHUB_REPO = "tim4431/Basler_Beam_Profiler"
RELEASE_API = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
RELEASE_ASSET = "pylon_camera.exe"
USER_AGENT = "beam-profiler-updater"

FIRST_CHECK_S = 15.0  # after start-up: let the camera open first
CHECK_INTERVAL_S = 6 * 3600.0  # the app often runs for days
GIT_TIMEOUT_S = 60
PIP_TIMEOUT_S = 900
DOWNLOAD_TIMEOUT_S = 60  # per read, not for the whole file

# Set across the re-exec after an update: the version it came from (a commit
# or a tag), so the new process can say what changed - and does not try again.
ENV_UPDATED_FROM = "BEAM_PROFILER_UPDATED_FROM"
ENV_SWITCH = "BEAM_PROFILER_UPDATE"


class UpdateError(Exception):
    """An update step that failed; the message is for the log and the toast."""


@dataclass(frozen=True)
class Status:
    """What the updater knows, for the status bar chip, toasts and the API.

    state: "off" (disabled), "unsupported" (neither a checkout nor a release
    exe), "idle" (not checked yet), "checking", "current", "ready" (an update
    is in place for the next start), "blocked" (one exists, but this checkout
    has local changes or commits), "error" (the check failed)."""

    state: str
    kind: str = ""  # "git" | "exe" | ""
    current: str = ""  # running version: short commit or release tag
    latest: str = ""  # what the next start will run, when state is "ready"
    detail: str = ""  # one human-readable line

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
#  git checkout
# --------------------------------------------------------------------------- #
_GITHUB_SSH = re.compile(r"^(?:ssh://)?git@github\.com[:/](.+?)(?:\.git)?/?$")


class GitInstall:
    """A git checkout that follows its branch's upstream by fast-forwarding."""

    kind = "git"

    def __init__(self, root: str, pip=None):
        self.root = root
        self._pip = pip or self._pip_install

    @staticmethod
    def detect(root: str) -> "GitInstall | None":
        if not os.path.exists(os.path.join(root, ".git")):
            return None
        try:
            subprocess.run(["git", "--version"], capture_output=True, timeout=10,
                           stdin=subprocess.DEVNULL)
        except (OSError, subprocess.SubprocessError):
            return None
        return GitInstall(root)

    def git(self, *args, check: bool = True, timeout: float = GIT_TIMEOUT_S) -> str:
        env = dict(os.environ, GIT_TERMINAL_PROMPT="0", LC_ALL="C")
        # Never prompt for an ssh passphrase: from a terminal that would block
        # this thread on stdin; from the desktop icon it would just hang.
        env.setdefault("GIT_SSH_COMMAND", "ssh -o BatchMode=yes -o ConnectTimeout=15")
        try:
            r = subprocess.run(["git", "-C", self.root, *args], capture_output=True,
                               text=True, timeout=timeout, env=env,
                               stdin=subprocess.DEVNULL)
        except (OSError, subprocess.SubprocessError) as e:
            raise UpdateError(f"git {args[0]}: {e}") from e
        if check and r.returncode:
            msg = (r.stderr or r.stdout).strip().splitlines()
            raise UpdateError(f"git {args[0]}: {msg[-1] if msg else 'failed'}")
        return r.stdout.strip()

    def version(self) -> str:
        try:
            return self.git("rev-parse", "--short", "HEAD")
        except UpdateError:
            return ""

    def _upstream(self) -> tuple[str, str, str]:
        """(branch, remote, upstream ref) - or UpdateError when this checkout
        does not follow anything."""
        branch = self.git("symbolic-ref", "--short", "-q", "HEAD", check=False)
        if not branch:
            raise UpdateError("detached HEAD: not following a branch")
        upstream = self.git("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}",
                            check=False)
        remote = self.git("config", f"branch.{branch}.remote", check=False)
        if not upstream or not remote or remote == ".":
            raise UpdateError(f"branch {branch} has no remote upstream")
        return branch, remote, upstream

    def fetch(self):
        branch, remote, upstream = self._upstream()
        merge = self.git("config", f"branch.{branch}.merge")  # refs/heads/main
        try:
            self.git("fetch", "--quiet", "--no-tags", remote, merge)
            return
        except UpdateError as e:
            ssh_error = e
        # No ssh key at hand (desktop session without an agent): a public
        # GitHub repo can be read over https as well.
        url = self.git("remote", "get-url", remote, check=False)
        match = _GITHUB_SSH.match(url)
        if not match:
            raise ssh_error
        tracking = "refs/remotes/" + upstream
        try:
            self.git("fetch", "--quiet", "--no-tags",
                     f"https://github.com/{match.group(1)}.git", f"+{merge}:{tracking}")
        except UpdateError:
            raise ssh_error from None

    def check(self, fetch: bool = True) -> Status:
        current = self.version()
        try:
            if fetch:
                self.fetch()
            _branch, _remote, upstream = self._upstream()
            behind = int(self.git("rev-list", "--count", "HEAD..@{u}"))
            ahead = int(self.git("rev-list", "--count", "@{u}..HEAD"))
            latest = self.git("rev-parse", "--short", "@{u}")
            dirty = self.git("status", "--porcelain", "--untracked-files=no")
        except (UpdateError, ValueError) as e:
            return Status("error", self.kind, current, detail=str(e))
        if not behind:
            return Status("current", self.kind, current, detail=f"up to date with {upstream}")
        news = f"{behind} new commit{'s' * (behind != 1)} on {upstream}"
        if ahead:
            return Status("blocked", self.kind, current, latest,
                          f"{news}, but {ahead} local commit{'s' * (ahead != 1)} "
                          f"{'are' if ahead != 1 else 'is'} not on it: merge by hand")
        if dirty:
            return Status("blocked", self.kind, current, latest,
                          f"{news}, but tracked files have local changes: "
                          f"commit or stash them, or pull by hand")
        return Status("ready", self.kind, current, latest, news)

    def apply(self) -> str | None:
        """Fast-forward to the fetched upstream (no network).  Returns the
        commit it came from, or None when there was nothing to do / it was not
        safe.  Raises UpdateError when the update failed (and was undone)."""
        status = self.check(fetch=False)
        if status.state != "ready":
            if status.state in ("blocked", "error"):
                logging.info(f"update: not updating - {status.detail}")
            return None
        old = self.git("rev-parse", "HEAD")
        logging.info(f"update: {status.detail}; fast-forwarding")
        self.git("merge", "--ff-only", "--quiet", "@{u}")
        changed = self.git("diff", "--name-only", old, "HEAD", "--", "requirements.txt")
        if changed:
            logging.info("update: requirements.txt changed; installing")
            try:
                self._pip()
            except UpdateError as e:
                self.git("reset", "--quiet", "--keep", old)
                raise UpdateError(f"{e} - stayed on {old[:7]}; run ./setup.sh "
                                  f"(or pip install -r requirements.txt)") from None
        return old

    def _pip_install(self):
        cmd = [sys.executable, "-m", "pip", "install", "--quiet",
               "--disable-pip-version-check", "-r", "requirements.txt"]
        try:
            r = subprocess.run(cmd, cwd=self.root, capture_output=True, text=True,
                               timeout=PIP_TIMEOUT_S, stdin=subprocess.DEVNULL)
        except (OSError, subprocess.SubprocessError) as e:
            raise UpdateError(f"pip: {e}") from e
        if r.returncode:
            lines = (r.stderr or r.stdout).strip().splitlines()
            raise UpdateError(f"pip install failed: {lines[-1] if lines else r.returncode}")

    def describe_since(self, old: str) -> str:
        try:
            n = int(self.git("rev-list", "--count", f"{old}..HEAD"))
            subject = self.git("log", "-1", "--format=%s")
        except (UpdateError, ValueError):
            return f"Updated to {self.version()}"
        return f"Updated to {self.version()} ({n} commit{'s' * (n != 1)}): {subject}"


# --------------------------------------------------------------------------- #
#  release exe
# --------------------------------------------------------------------------- #
def version_key(tag: str) -> tuple[int, ...]:
    """Release tags are vYYYY.MM.DD.HHMM: compare them as numbers."""
    return tuple(int(n) for n in re.findall(r"\d+", tag or ""))


class ExeInstall:
    """The one-file exe from a GitHub release, replaced by the newest one.

    A verified download waits as ``<exe>.<tag>.new`` - the tag in the name, so
    a start never swaps in a build older than the one running (say, after the
    exe was replaced by hand)."""

    kind = "exe"

    def __init__(self, exe: str, release: str, urlopen=None):
        self.exe = exe
        self.release = release  # what is running
        self._urlopen = urlopen or urllib.request.urlopen

    @staticmethod
    def detect() -> "ExeInstall | None":
        if getattr(sys, "frozen", False) and sys.platform == "win32" and RELEASE:
            return ExeInstall(sys.executable, RELEASE)
        return None

    def version(self) -> str:
        return self.release

    def staged_path(self, tag: str) -> str:
        return f"{self.exe}.{tag}.new"

    def _staged(self) -> list[tuple[str, str]]:
        """(tag, path) of the downloads newer than what runs, newest first."""
        prefix, found = self.exe + ".", []
        for path in glob.glob(glob.escape(self.exe) + ".*.new"):
            tag = path[len(prefix):-len(".new")]
            found.append((version_key(tag), tag, path))
        mine = version_key(self.release)
        return [(tag, path) for key, tag, path in sorted(found, reverse=True) if key > mine]

    @property
    def staged(self) -> str | None:
        """The newest downloaded release, or None."""
        newest = self._staged()
        return newest[0][0] if newest else None

    def cleanup(self):
        """Delete what the last swap left (the replaced exe), a partial download,
        and downloads superseded by a newer one or by the running exe."""
        keep = self._staged()[:1]
        paths = glob.glob(glob.escape(self.exe) + ".old*") + [self.exe + ".download"]
        paths += [p for p in glob.glob(glob.escape(self.exe) + ".*.new")
                  if p not in {path for _, path in keep}]
        for path in paths:
            try:
                os.remove(path)
            except FileNotFoundError:
                pass
            except OSError as e:  # the old copy may still be exiting
                logging.debug(f"update: could not remove {path}: {e}")

    def _get(self, url: str, accept: str):
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT,
                                                   "Accept": accept})
        return self._urlopen(req, timeout=DOWNLOAD_TIMEOUT_S)

    def check(self, fetch: bool = True) -> Status:
        self.cleanup()
        staged = self.staged
        newest = staged or self.release
        try:
            with self._get(RELEASE_API, "application/vnd.github+json") as r:
                info = json.load(r)
            tag = info["tag_name"]
            if version_key(tag) <= version_key(newest):
                if staged:
                    return Status("ready", self.kind, self.release, staged,
                                  f"{staged} downloaded; it runs from the next start")
                return Status("current", self.kind, self.release,
                              detail=f"{self.release} is the latest release")
            asset = next((a for a in info.get("assets", ())
                          if a.get("name") == RELEASE_ASSET), None)
            if asset is None:
                raise UpdateError(f"release {tag} has no {RELEASE_ASSET}")
            logging.info(f"update: downloading {tag} ({asset.get('size', 0) / 1e6:.0f} MB)")
            self._download(asset, tag)
        except UpdateError as e:
            return Status("error", self.kind, self.release, detail=str(e))
        except Exception as e:  # network, JSON, file system
            return Status("error", self.kind, self.release,
                          detail=f"{type(e).__name__}: {e}")
        return Status("ready", self.kind, self.release, tag,
                      f"{tag} downloaded; it runs from the next start")

    def _download(self, asset: dict, tag: str):
        part = self.exe + ".download"
        digest = hashlib.sha256()
        size = 0
        with self._get(asset["browser_download_url"], "application/octet-stream") as r, \
                open(part, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
                digest.update(chunk)
                size += len(chunk)
        expected = asset.get("digest") or ""
        if size != asset.get("size", size) or (
                expected.startswith("sha256:") and expected[7:] != digest.hexdigest()):
            os.remove(part)
            raise UpdateError(f"download of {asset.get('name')} is corrupt "
                              f"({size} bytes, sha256 {digest.hexdigest()[:12]}...)")
        os.replace(part, self.staged_path(tag))

    def swap(self) -> bool:
        """Rename the downloaded build into place.  Only right before this
        process is replaced: from here on its own lazy imports would read the
        new file.  True when it swapped; OSError when it could not (undone)."""
        staged = self._staged()
        if not staged:
            return False
        new = staged[0][1]
        old = self.exe + ".old"
        try:
            os.remove(old)
        except FileNotFoundError:
            pass
        except OSError:  # an older copy is still exiting from it
            old = f"{self.exe}.old{int(time.time())}"
        os.replace(self.exe, old)  # renaming a running exe is allowed
        try:
            os.replace(new, self.exe)
        except OSError:
            os.replace(old, self.exe)
            raise
        return True


# --------------------------------------------------------------------------- #
#  other copies of the app
# --------------------------------------------------------------------------- #
def _instances_dir() -> str:
    return os.path.join(APP_DIR, "data", ".instances")


def _try_lock(f) -> bool:
    """Non-blocking exclusive lock on an open file (released when it closes)."""
    try:
        if os.name == "nt":
            import msvcrt

            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except OSError:
        return False


_instance_file = None  # held open (and locked) for the life of the process


def register_instance():
    """Mark this process as a running copy of the app until it exits.  The
    lock goes with the file descriptor, so an exec drops it and the new image
    takes it again under the same pid."""
    global _instance_file
    try:
        os.makedirs(_instances_dir(), exist_ok=True)
        f = open(os.path.join(_instances_dir(), str(os.getpid())), "a+")
    except OSError as e:
        logging.debug(f"update: cannot register this instance: {e}")
        return
    if _try_lock(f):
        _instance_file = f
    else:
        f.close()


def others_running() -> list[int]:
    """Pids of other copies of the app from this folder that are running.
    Files nobody holds a lock on are left by copies that ended; old ones are
    removed (not new ones: a copy may be between creating and locking)."""
    found = []
    try:
        names = os.listdir(_instances_dir())
    except OSError:
        return found
    for name in names:
        if name == str(os.getpid()) or not name.isdigit():
            continue
        path = os.path.join(_instances_dir(), name)
        try:
            with open(path, "a+") as f:
                alive = not _try_lock(f)
        except OSError:
            alive = True
        if alive:
            found.append(int(name))
            continue
        try:
            if time.time() - os.path.getmtime(path) > 60:
                os.remove(path)
        except OSError:
            pass
    return found


# --------------------------------------------------------------------------- #
#  the process-wide updater
# --------------------------------------------------------------------------- #
_lock = threading.Lock()
_status = Status("off")
_install = None
_wake = threading.Event()
_manual = False  # the user asked (U): report the result even if nothing changed
_thread = None


def status() -> Status:
    return _status


def _set(st: Status) -> Status:
    global _status
    with _lock:
        _status = st
    return st


def disabled_reason(cli_off: bool = False) -> str | None:
    """Why auto-update is off, or None when it is on."""
    if cli_off:
        return "turned off with --no-update"
    if os.environ.get(ENV_SWITCH, "").strip().lower() in ("0", "off", "no", "false"):
        return f"turned off with {ENV_SWITCH}=0"
    from . import uiconfig

    if not uiconfig.active().auto_update:
        return 'turned off in ui_config.json ("auto_update": false)'
    return None


def detect():
    """The install this process runs from, if it can update itself."""
    return ExeInstall.detect() or (None if getattr(sys, "frozen", False)
                                   else GitInstall.detect(APP_DIR))


def apply_at_startup(cli_off: bool = False) -> None:
    """Called first thing in main(): clean up after the last update, and
    install one fetched earlier - which re-executes the process."""
    global _install
    reason = disabled_reason(cli_off)
    _install = detect()
    if _install is None:
        _set(Status("unsupported", detail="neither a git checkout nor a release exe; "
                                          "update by hand"))
        return
    if reason:
        _set(Status("off", _install.kind, _install.version(), detail=reason))
        return
    _set(Status("idle", _install.kind, _install.version()))
    register_instance()
    if os.environ.get(ENV_UPDATED_FROM):
        return  # just updated: this is the new code
    others = others_running()
    if others:
        logging.info(f"update: another copy of the app is running (pid "
                     f"{', '.join(map(str, others))}); not installing anything under it")
        return
    try:
        if isinstance(_install, ExeInstall):
            _install.cleanup()
            old = _install.release if _install.swap() else None
        else:
            old = _install.apply()
    except (UpdateError, OSError) as e:
        logging.error(f"update failed: {e}")
        _set(Status("error", _install.kind, _install.version(), detail=str(e)))
        return
    if old:
        relaunch({ENV_UPDATED_FROM: old})


def just_updated() -> str | None:
    """'Updated to ...' when this process is the first run of a new version
    (once: the variable is cleared so a later relaunch does not repeat it)."""
    old = os.environ.pop(ENV_UPDATED_FROM, None)
    if not old or _install is None:
        return None
    if isinstance(_install, GitInstall):
        return _install.describe_since(old)
    return f"Updated to {_install.release} (from {old})"


def start(notify) -> None:
    """Check in the background now and then; `notify(message, level)` (the
    viewer's toast, safe from any thread) hears when an update is ready."""
    global _thread
    if _install is None or _status.state in ("off", "unsupported") or _thread:
        return
    _thread = threading.Thread(target=_loop, args=(_install, notify), daemon=True,
                               name="updater")
    _thread.start()


def check_now() -> None:
    """Check at once (U when no update is waiting); the result is toasted."""
    global _manual
    _manual = True
    _wake.set()


def _loop(install, notify):
    global _manual
    _wake.wait(FIRST_CHECK_S)
    announced = None
    while True:
        _wake.clear()
        manual, _manual = _manual, False
        before = _status
        _set(Status("checking", install.kind, install.version(), before.latest))
        try:
            st = install.check()
        except Exception as e:  # a bug here must not kill the thread
            st = Status("error", install.kind, install.version(), detail=str(e))
        _set(st)
        if st.state == "ready" and st.latest != announced:
            announced = st.latest
            logging.info(f"update: {st.detail}")
            notify(f"Update ready: {st.detail}. Press U to restart into it", "info")
        elif manual:
            level = {"error": "error", "blocked": "warn"}.get(st.state, "info")
            notify({"current": "Up to date", "ready": f"Update ready: {st.detail}"}
                   .get(st.state, f"Update: {st.detail}"), level)
        if st.state == "error":
            logging.info(f"update check failed: {st.detail}")
        elif st.state == "blocked" and before.state != "blocked":
            logging.info(f"update: {st.detail}")
        _wake.wait(CHECK_INTERVAL_S)


def update_now(cli_off: bool = False) -> int:
    """--update: check, install, report, exit code 0 when up to date after."""
    reason = disabled_reason(cli_off)
    install = detect()
    if install is None:
        print("This copy is neither a git checkout nor a release exe: update by hand.")
        return 1
    if reason:
        print(f"Auto-update is {reason}; checking anyway.")
    print(f"Running {install.kind} {install.version() or '?'}; checking...")
    st = install.check()
    print(f"{st.state}: {st.detail}")
    if st.state != "ready":
        return 0 if st.state == "current" else 1
    others = others_running()
    if others:
        print(f"Not installed: the app is running (pid {', '.join(map(str, others))}). "
              "Close it, or press U in it.")
        return 1
    try:
        if isinstance(install, ExeInstall):
            install.swap()  # this process exits right after
            print(f"Installed {st.latest}; it runs from the next start.")
        elif old := install.apply():
            print(install.describe_since(old))
    except (UpdateError, OSError) as e:
        print(f"update failed: {e}")
        return 1
    return 0


def restart():
    """U, after the viewer has closed: start the update that is ready.  A
    checkout merges in the new process's `apply_at_startup`; the exe is swapped
    here, the last thing this process does."""
    env = {}
    if isinstance(_install, ExeInstall) and not others_running():
        try:
            if _install.swap():
                env[ENV_UPDATED_FROM] = _install.release
        except OSError as e:
            logging.error(f"update: could not replace {_install.exe}: {e}")
    relaunch(env)


def relaunch(extra_env: dict | None = None):
    """Replace this process with a fresh start of the same command line."""
    env = dict(os.environ, **(extra_env or {}))
    if getattr(sys, "frozen", False):
        argv = [sys.executable, *sys.argv[1:]]
        # PyInstaller >= 6.9: start a new instance, not a child of this one
        # (which would reuse this one's soon-deleted unpack directory).
        env["PYINSTALLER_RESET_ENVIRONMENT"] = "1"
    else:
        argv = [sys.executable, *sys.orig_argv[1:]]
    logging.info("restarting: " + " ".join(argv))
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:
            pass
    if os.name == "nt":  # no real exec on Windows
        subprocess.Popen(argv, env=env, close_fds=True)
        os._exit(0)
    os.execve(sys.executable, argv, env)
