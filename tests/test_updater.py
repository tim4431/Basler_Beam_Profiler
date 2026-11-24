"""Auto-update without the network: real git repositories in a temp dir for
checkouts, and a fake GitHub API for the release exe."""

import hashlib
import io
import json
import os
import pathlib
import subprocess
import threading

import pytest

from beam_profiler import uiconfig, updater
from beam_profiler.updater import ExeInstall, GitInstall, Status, UpdateError

@pytest.fixture(autouse=True)
def fresh_updater(monkeypatch):
    """Module state starts empty in every test; no test may really restart."""
    monkeypatch.setattr(updater, "_status", Status("off"))
    monkeypatch.setattr(updater, "_install", None)
    monkeypatch.setattr(updater, "_thread", None)
    monkeypatch.setattr(updater, "_wake", threading.Event())
    monkeypatch.setattr(updater, "_instance_file", None)
    monkeypatch.delenv(updater.ENV_UPDATED_FROM, raising=False)
    monkeypatch.delenv(updater.ENV_SWITCH, raising=False)

    def no_relaunch(extra_env=None):
        raise AssertionError("relaunch() in a test")

    monkeypatch.setattr(updater, "relaunch", no_relaunch)
    yield
    if updater._instance_file:
        updater._instance_file.close()


# --------------------------------------------------------------------------- #
#  git checkouts
# --------------------------------------------------------------------------- #
def git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True,
                          text=True).stdout.strip()


def commit(repo, name, text, message=None):
    with open(os.path.join(repo, name), "w") as f:
        f.write(text)
    git(repo, "add", name)
    git(repo, "commit", "-q", "-m", message or f"change {name}")
    return git(repo, "rev-parse", "HEAD")


@pytest.fixture
def repos(tmp_path, monkeypatch):
    """origin (bare), `app` (the install, tracking origin/main) and `dev`
    (where new versions are pushed from)."""
    try:
        subprocess.run(["git", "--version"], capture_output=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("git not installed")
    empty = tmp_path / "gitconfig"
    empty.write_text("")
    for key, value in {
        "GIT_CONFIG_GLOBAL": str(empty), "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid",
        "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.invalid",
    }.items():
        monkeypatch.setenv(key, value)
    origin, app, dev = (str(tmp_path / n) for n in ("origin.git", "app", "dev"))
    git(tmp_path, "init", "-q", "--bare", "-b", "main", origin)
    git(tmp_path, "clone", "-q", origin, dev)
    git(dev, "checkout", "-q", "-b", "main")
    commit(dev, "requirements.txt", "numpy\n")
    commit(dev, "app.py", "v1\n")
    git(dev, "push", "-q", "-u", "origin", "main")
    git(tmp_path, "clone", "-q", origin, app)
    return origin, app, dev


def read(repo, name):
    with open(os.path.join(repo, name)) as f:
        return f.read()


def test_nothing_new_is_current(repos):
    _, app, _ = repos
    st = GitInstall(app).check()
    assert st.state == "current" and st.kind == "git"
    assert st.current == git(app, "rev-parse", "--short", "HEAD")
    assert GitInstall(app).apply() is None


def test_a_pushed_commit_is_fetched_then_fast_forwarded(repos):
    _, app, dev = repos
    old = git(app, "rev-parse", "HEAD")
    new = commit(dev, "app.py", "v2\n", "Faster fit")
    git(dev, "push", "-q")

    inst = GitInstall(app, pip=lambda: pytest.fail("requirements did not change"))
    assert inst.check(fetch=False).state == "current"  # nothing fetched yet
    st = inst.check()
    assert st.state == "ready" and st.latest == new[:7]
    assert "1 new commit on origin/main" in st.detail
    assert read(app, "app.py") == "v1\n"  # a check never touches the tree

    assert inst.apply() == old
    assert read(app, "app.py") == "v2\n"
    assert git(app, "rev-parse", "HEAD") == new
    assert inst.describe_since(old) == f"Updated to {new[:7]} (1 commit): Faster fit"
    assert inst.check(fetch=False).state == "current"


def test_local_changes_block_the_update(repos):
    _, app, dev = repos
    commit(dev, "app.py", "v2\n")
    git(dev, "push", "-q")
    with open(os.path.join(app, "app.py"), "a") as f:
        f.write("# tweak\n")
    inst = GitInstall(app)
    st = inst.check()
    assert st.state == "blocked" and "local changes" in st.detail
    assert inst.apply() is None
    assert read(app, "app.py") == "v1\n# tweak\n"


def test_untracked_files_do_not_block(repos):
    _, app, dev = repos
    commit(dev, "app.py", "v2\n")
    git(dev, "push", "-q")
    with open(os.path.join(app, "notes.txt"), "w") as f:
        f.write("mine\n")
    inst = GitInstall(app)
    assert inst.check().state == "ready"
    assert inst.apply()
    assert read(app, "app.py") == "v2\n" and read(app, "notes.txt") == "mine\n"


def test_local_commits_are_never_merged_over(repos):
    _, app, dev = repos
    commit(dev, "app.py", "v2\n")
    git(dev, "push", "-q")
    local = commit(app, "other.py", "x\n")
    inst = GitInstall(app)
    st = inst.check()
    assert st.state == "blocked" and "1 local commit is not on it" in st.detail
    assert inst.apply() is None
    assert git(app, "rev-parse", "HEAD") == local


def test_detached_head_or_no_upstream_is_an_error_not_an_exception(repos):
    _, app, _ = repos
    git(app, "checkout", "-q", "--detach")
    st = GitInstall(app).check()
    assert st.state == "error" and "detached" in st.detail
    git(app, "checkout", "-q", "-b", "local-only")
    st = GitInstall(app).check()
    assert st.state == "error" and "no remote upstream" in st.detail


def test_changed_requirements_are_installed(repos):
    _, app, dev = repos
    commit(dev, "requirements.txt", "numpy\nnewdep\n")
    git(dev, "push", "-q")
    calls = []
    inst = GitInstall(app, pip=lambda: calls.append(read(app, "requirements.txt")))
    inst.check()
    assert inst.apply()
    assert calls == ["numpy\nnewdep\n"]  # pip saw the new file


def test_failed_pip_rolls_the_update_back(repos):
    _, app, dev = repos
    old = git(app, "rev-parse", "HEAD")
    commit(dev, "requirements.txt", "numpy\nnewdep\n")
    git(dev, "push", "-q")

    def broken_pip():
        raise UpdateError("pip install failed: no network")

    inst = GitInstall(app, pip=broken_pip)
    inst.check()
    with pytest.raises(UpdateError, match="stayed on"):
        inst.apply()
    assert git(app, "rev-parse", "HEAD") == old
    assert read(app, "requirements.txt") == "numpy\n"
    assert inst.check(fetch=False).state == "ready"  # tried again next start


def test_a_failed_fetch_is_an_error_status(repos):
    origin, app, _ = repos
    git(app, "remote", "set-url", "origin", origin + "-gone")
    st = GitInstall(app).check()
    assert st.state == "error" and "fetch" in st.detail


@pytest.mark.parametrize("url, path", [
    ("git@github.com:tim4431/Basler_Beam_Profiler.git", "tim4431/Basler_Beam_Profiler"),
    ("ssh://git@github.com/owner/repo", "owner/repo"),
    ("https://github.com/owner/repo.git", None),
    ("git@gitlab.com:owner/repo.git", None),
])
def test_github_ssh_remotes_have_an_https_fallback(url, path):
    match = updater._GITHUB_SSH.match(url)
    assert (match.group(1) if match else None) == path


def test_startup_installs_a_fetched_update_and_restarts(repos, monkeypatch):
    _, app, dev = repos
    old = git(app, "rev-parse", "HEAD")
    commit(dev, "app.py", "v2\n", "New feature")
    git(dev, "push", "-q")
    GitInstall(app).fetch()  # what the last session's background check did
    monkeypatch.setattr(updater, "APP_DIR", app)
    restarts = []
    monkeypatch.setattr(updater, "relaunch", restarts.append)

    updater.apply_at_startup()
    assert restarts == [{updater.ENV_UPDATED_FROM: old}]
    assert read(app, "app.py") == "v2\n"

    # the re-executed process: says what changed, does not apply again
    monkeypatch.setenv(updater.ENV_UPDATED_FROM, old)
    updater.apply_at_startup()
    assert len(restarts) == 1
    assert updater.just_updated().endswith("(1 commit): New feature")
    assert updater.just_updated() is None  # once
    assert updater.status().state == "idle"


@pytest.mark.parametrize("how", ["cli", "env", "config"])
def test_auto_update_can_be_turned_off(repos, monkeypatch, how):
    _, app, dev = repos
    commit(dev, "app.py", "v2\n")
    git(dev, "push", "-q")
    GitInstall(app).fetch()
    monkeypatch.setattr(updater, "APP_DIR", app)
    if how == "env":
        monkeypatch.setenv(updater.ENV_SWITCH, "0")
    elif how == "config":
        uiconfig.use(uiconfig.UIConfig(auto_update=False))
    updater.apply_at_startup(cli_off=how == "cli")
    st = updater.status()
    assert st.state == "off" and st.kind == "git" and "turned off" in st.detail
    assert read(app, "app.py") == "v1\n"
    updater.start(lambda *a: pytest.fail("no checks when off"))
    assert updater._thread is None


def test_a_plain_folder_is_unsupported(tmp_path, monkeypatch):
    monkeypatch.setattr(updater, "APP_DIR", str(tmp_path))
    updater.apply_at_startup()
    assert updater.status().state == "unsupported"


def test_update_now_from_the_command_line(repos, monkeypatch, capsys):
    _, app, dev = repos
    commit(dev, "app.py", "v2\n", "From the CLI")
    git(dev, "push", "-q")
    monkeypatch.setattr(updater, "APP_DIR", app)
    assert updater.update_now() == 0
    out = capsys.readouterr().out
    assert "ready: 1 new commit" in out and "From the CLI" in out
    assert read(app, "app.py") == "v2\n"
    assert updater.update_now() == 0
    assert "current: up to date" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
#  release exe
# --------------------------------------------------------------------------- #
NEW_EXE = b"MZ new build " * 1000


class FakeGitHub:
    def __init__(self, tag, payload=NEW_EXE, digest=None, size=None):
        self.tag, self.payload, self.requests = tag, payload, []
        self.digest = digest or "sha256:" + hashlib.sha256(payload).hexdigest()
        self.size = len(payload) if size is None else size

    def __call__(self, req, timeout=None):
        self.requests.append(req.full_url)
        assert req.get_header("User-agent") == updater.USER_AGENT
        if req.full_url == updater.RELEASE_API:
            return io.BytesIO(json.dumps({"tag_name": self.tag, "assets": [
                {"name": "notes.txt", "browser_download_url": "x", "size": 1},
                {"name": updater.RELEASE_ASSET, "size": self.size, "digest": self.digest,
                 "browser_download_url": "https://example.invalid/pylon_camera.exe"},
            ]}).encode())
        return io.BytesIO(self.payload)


@pytest.fixture
def exe(tmp_path):
    path = tmp_path / "pylon_camera.exe"
    path.write_bytes(b"MZ old build")
    return str(path)


@pytest.mark.parametrize("a, b", [
    ("v2026.11.05.0048", "v2026.11.05.0049"),
    ("v2026.11.05.2359", "v2026.11.06.0000"),
    ("v2026.08.22.0737", "v2026.11.04.0249"),
])
def test_release_tags_compare_by_date(a, b):
    assert updater.version_key(a) < updater.version_key(b)


def test_a_newer_release_is_downloaded_then_swapped_in_at_the_next_start(exe):
    gh = FakeGitHub("v2026.12.01.1200")
    inst = ExeInstall(exe, "v2026.11.05.0048", urlopen=gh)
    st = inst.check()
    assert st.state == "ready" and st.latest == "v2026.12.01.1200"
    assert st.current == "v2026.11.05.0048"
    staged = exe + ".v2026.12.01.1200.new"
    assert open(staged, "rb").read() == NEW_EXE
    # the running exe is untouched: its lazy imports read this file by path
    assert open(exe, "rb").read() == b"MZ old build"
    assert not os.path.exists(exe + ".download")

    # later checks see it is downloaded: no second download
    n = len(gh.requests)
    assert inst.check().state == "ready"
    assert len(gh.requests) == n + 1  # the API only

    assert inst.swap()  # the next start
    assert open(exe, "rb").read() == NEW_EXE
    assert open(exe + ".old", "rb").read() == b"MZ old build"
    assert not os.path.exists(staged)
    assert not inst.swap()  # nothing left to swap

    ExeInstall(exe, "v2026.12.01.1200").cleanup()  # the start after
    assert sorted(os.listdir(os.path.dirname(exe))) == ["pylon_camera.exe"]


def test_a_download_older_than_the_running_exe_is_never_swapped_in(exe):
    """Say v2026.12 was downloaded, then the exe was replaced by hand with a
    newer one: the stale download must not downgrade it."""
    with open(exe + ".v2026.12.01.1200.new", "wb") as f:
        f.write(NEW_EXE)
    inst = ExeInstall(exe, "v2027.01.01.0000")
    assert inst.staged is None
    assert not inst.swap()
    assert open(exe, "rb").read() == b"MZ old build"
    inst.cleanup()
    assert sorted(os.listdir(os.path.dirname(exe))) == ["pylon_camera.exe"]


def test_exe_startup_swaps_and_starts_the_new_build(exe, monkeypatch, tmp_path):
    monkeypatch.setattr(updater, "APP_DIR", str(tmp_path))
    inst = ExeInstall(exe, "v2026.11.05.0048")
    monkeypatch.setattr(ExeInstall, "detect", staticmethod(lambda: inst))
    with open(inst.staged_path("v2026.12.01.1200"), "wb") as f:
        f.write(NEW_EXE)
    restarts = []
    monkeypatch.setattr(updater, "relaunch", restarts.append)
    updater.apply_at_startup()
    assert restarts == [{updater.ENV_UPDATED_FROM: "v2026.11.05.0048"}]
    assert open(exe, "rb").read() == NEW_EXE


def test_u_swaps_the_exe_as_the_last_thing_before_restarting(exe, monkeypatch):
    inst = ExeInstall(exe, "v2026.11.05.0048", urlopen=FakeGitHub("v2026.12.01.1200"))
    inst.check()
    monkeypatch.setattr(updater, "_install", inst)
    restarts = []
    monkeypatch.setattr(updater, "relaunch", restarts.append)
    monkeypatch.setattr(updater, "others_running", lambda: [4242])
    updater.restart()  # another copy appeared meanwhile: not under it
    assert restarts == [{}] and open(exe, "rb").read() == b"MZ old build"
    monkeypatch.setattr(updater, "others_running", lambda: [])
    updater.restart()
    assert restarts[-1] == {updater.ENV_UPDATED_FROM: "v2026.11.05.0048"}
    assert open(exe, "rb").read() == NEW_EXE


def test_the_same_release_is_current(exe):
    gh = FakeGitHub("v2026.11.05.0048")
    st = ExeInstall(exe, "v2026.11.05.0048", urlopen=gh).check()
    assert st.state == "current"
    assert gh.requests == [updater.RELEASE_API]
    assert open(exe, "rb").read() == b"MZ old build"


@pytest.mark.parametrize("bad", [dict(digest="sha256:" + "0" * 64), dict(size=5)])
def test_a_corrupt_download_is_thrown_away(exe, bad):
    inst = ExeInstall(exe, "v2026.11.05.0048", urlopen=FakeGitHub("v2026.12.01.1200", **bad))
    st = inst.check()
    assert st.state == "error" and "corrupt" in st.detail
    assert open(exe, "rb").read() == b"MZ old build"
    assert sorted(os.listdir(os.path.dirname(exe))) == ["pylon_camera.exe"]


def test_no_network_is_an_error_status(exe):
    def offline(req, timeout=None):
        raise OSError("Network is unreachable")

    st = ExeInstall(exe, "v2026.11.05.0048", urlopen=offline).check()
    assert st.state == "error" and "unreachable" in st.detail


# --------------------------------------------------------------------------- #
#  the background thread
# --------------------------------------------------------------------------- #
class FakeInstall:
    kind = "git"

    def __init__(self, *states):
        self.states = list(states)

    def version(self):
        return "abc1234"

    def check(self, fetch=True):
        state = self.states.pop(0) if len(self.states) > 1 else self.states[0]
        return Status(state, self.kind, "abc1234", "def5678" if state == "ready" else "",
                      "2 new commits on origin/main" if state == "ready" else "up to date")


def start_thread(monkeypatch, install, first_check_s=0):
    monkeypatch.setattr(updater, "FIRST_CHECK_S", first_check_s)
    monkeypatch.setattr(updater, "_install", install)
    monkeypatch.setattr(updater, "_status", Status("idle", "git", "abc1234"))
    heard, said = [], threading.Event()

    def notify(message, level="info"):
        heard.append((message, level))
        said.set()

    updater.start(notify)
    return heard, said


def test_the_thread_announces_a_ready_update_once(monkeypatch):
    heard, said = start_thread(monkeypatch, FakeInstall("ready"))
    assert said.wait(5), "the updater thread said nothing"
    assert heard == [("Update ready: 2 new commits on origin/main. "
                      "Press U to restart into it", "info")]
    assert updater.status().state == "ready"


def test_a_manual_check_reports_even_when_up_to_date(monkeypatch):
    heard, said = start_thread(monkeypatch, FakeInstall("current"), first_check_s=3600)
    updater.check_now()  # U: checks at once instead of in an hour
    assert said.wait(5), "the updater thread said nothing"
    assert heard == [("Up to date", "info")]
    assert updater.status().state == "current"


# --------------------------------------------------------------------------- #
#  other copies of the app
# --------------------------------------------------------------------------- #
def hold_instance(tmp_path, pid):
    """Pretend copy `pid` is running: its file, locked through its own fd."""
    folder = tmp_path / "data" / ".instances"
    folder.mkdir(parents=True, exist_ok=True)
    f = open(folder / str(pid), "a+")
    assert updater._try_lock(f)
    return f


def test_running_copies_are_found_and_ended_ones_cleaned_up(tmp_path, monkeypatch):
    monkeypatch.setattr(updater, "APP_DIR", str(tmp_path))
    assert updater.others_running() == []
    updater.register_instance()
    assert updater.others_running() == []  # not this one
    other = hold_instance(tmp_path, 999999)
    assert updater.others_running() == [999999]
    other.close()  # that copy ended (or crashed: the OS drops the lock either way)
    path = tmp_path / "data" / ".instances" / "999999"
    assert updater.others_running() == [] and path.exists()  # too new to delete
    os.utime(path, (0, 0))
    assert updater.others_running() == [] and not path.exists()


def test_nothing_is_installed_under_another_running_copy(repos, monkeypatch, tmp_path):
    _, app, dev = repos
    commit(dev, "app.py", "v2\n")
    git(dev, "push", "-q")
    GitInstall(app).fetch()
    monkeypatch.setattr(updater, "APP_DIR", app)
    other = hold_instance(pathlib.Path(app), 999999)
    try:
        updater.apply_at_startup()  # would relaunch (and fail the test) if it merged
        assert read(app, "app.py") == "v1\n"
        assert updater.update_now() == 1
        assert read(app, "app.py") == "v1\n"
    finally:
        other.close()
