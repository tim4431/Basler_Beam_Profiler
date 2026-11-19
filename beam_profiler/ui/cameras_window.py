"""Cameras window (`k`, or the Cameras button): every camera the app can see,
updated live, with a click to switch, and a form for the two settings a newly
added camera needs - a label and its pixel size - saved to camera_config.yaml.

A Toplevel on the shared Tk thread (see `tkhost`).  It never touches a Camera:
it reads `manager.snapshot()` twice a second and posts requests that the main
loop carries out, so a slow camera cannot freeze it and it cannot race the
grab loop.  Saving edits camera_config.yaml in place (comments survive, see
`cameraconfig.save_entry`) and then asks the main loop to reload it.
"""

from __future__ import annotations

import os
import tkinter as tk
from tkinter import ttk

from .. import cameraconfig
from . import tkhost
from .tkhost import BAD, GOOD, MUTED, WARN

REFRESH_MS = 500
COLUMNS = (
    ("state", "Status", 120),
    ("label", "Camera", 190),
    ("model", "Model", 170),
    ("serial", "Serial", 110),
    ("source", "Connection", 150),
)
_window: CamerasWindow | None = None


def open_cameras(manager):
    tkhost.call(lambda root: _open(root, manager))


def _open(root, manager):
    global _window
    if _window is None or not _window.exists():
        _window = CamerasWindow(manager)
        _window.build(root)
    else:
        _window.show()


def close_cameras():
    tkhost.call(lambda _root: _window.close() if _window is not None else None)


def is_open() -> bool:
    w = _window
    return bool(w is not None and w.visible)


class CamerasWindow:
    def __init__(self, manager):
        self.manager = manager
        self.visible = False
        self.win: tk.Toplevel | None = None
        self._rows: dict[str, dict] = {}
        self._form_uid: str | None = None  # whose values the form holds
        self._status_until = 0

    # ------------------------------------------------------------------ #
    def build(self, master: tk.Misc):
        self.win = win = tk.Toplevel(master)
        win.title("Cameras")
        win.minsize(780, 600)
        win.configure(background=tkhost.BG)
        win.protocol("WM_DELETE_WINDOW", self.close)
        win.bind("<Escape>", lambda _e: self.close())
        win.bind("<F5>", lambda _e: self._rescan())

        header = ttk.Frame(win, padding=(16, 14, 16, 6))
        header.pack(fill="x")
        ttk.Label(header, text="Cameras", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            header,
            text="Plug a camera in and it shows up here within a few seconds. "
                 "Double-click one to switch to it.",
            style="Sub.TLabel",
        ).pack(anchor="w", pady=(2, 0))

        # --- the list ------------------------------------------------------
        table = ttk.Frame(win, padding=(16, 4, 16, 0))
        table.pack(fill="both", expand=True)
        self.tree = ttk.Treeview(
            table, columns=[c[0] for c in COLUMNS], show="headings",
            selectmode="browse", height=7,
        )
        for key, title, width in COLUMNS:
            self.tree.heading(key, text=title, anchor="w")
            self.tree.column(key, width=width, anchor="w", stretch=key in ("label", "model"))
        bar = ttk.Scrollbar(table, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=bar.set)
        self.tree.pack(side="left", fill="both", expand=True)
        bar.pack(side="right", fill="y")
        self.tree.tag_configure("active", foreground=GOOD)
        self.tree.tag_configure("connecting", foreground=WARN)
        self.tree.tag_configure("lost", foreground=BAD)
        self.tree.tag_configure("error", foreground=BAD)
        self.tree.tag_configure("absent", foreground=MUTED)
        self.tree.bind("<Double-1>", lambda _e: self._use())
        self.tree.bind("<Return>", lambda _e: self._use())
        self.tree.bind("<<TreeviewSelect>>", lambda _e: self._on_select())

        actions = ttk.Frame(win, padding=(16, 8, 16, 0))
        actions.pack(fill="x")
        self.use_btn = ttk.Button(
            actions, text="Switch to this camera", style="Accent.TButton", command=self._use
        )
        self.use_btn.pack(side="left")
        ttk.Button(actions, text="Rescan (F5)", command=self._rescan).pack(
            side="left", padx=(8, 0)
        )
        self.status = ttk.Label(actions, text="", style="Sub.TLabel", wraplength=380)
        self.status.pack(side="right")

        # --- details / settings for the selected camera --------------------
        card = ttk.LabelFrame(
            win, text=" Selected camera ", style="Card.TLabelframe", padding=(14, 10)
        )
        card.pack(fill="x", padx=16, pady=(12, 0))
        card.columnconfigure(1, weight=1)
        self.title = ttk.Label(card, text="", style="CardBold.TLabel")
        self.title.grid(row=0, column=0, columnspan=3, sticky="w")
        self.detail = ttk.Label(card, text="", style="CardSub.TLabel", justify="left")
        self.detail.grid(row=1, column=0, columnspan=3, sticky="w", pady=(2, 8))

        self.label_var = tk.StringVar()
        self.pixel_var = tk.StringVar()
        ttk.Label(card, text="Label", style="Card.TLabel").grid(row=2, column=0, sticky="w")
        ttk.Entry(card, textvariable=self.label_var).grid(
            row=2, column=1, columnspan=2, sticky="ew", padx=(12, 0), pady=3
        )
        ttk.Label(card, text="Pixel size", style="Card.TLabel").grid(row=3, column=0, sticky="w")
        pixel = ttk.Frame(card, style="Card.TFrame")
        pixel.grid(row=3, column=1, columnspan=2, sticky="ew", padx=(12, 0), pady=3)
        entry = ttk.Entry(pixel, textvariable=self.pixel_var, width=10, justify="right")
        entry.pack(side="left")
        entry.bind("<Return>", lambda _e: self._save())
        ttk.Label(pixel, text="µm per pixel  (from the sensor datasheet)",
                  style="CardSub.TLabel").pack(side="left", padx=(6, 0))
        self.scale_note = ttk.Label(card, text="", style="CardWarn.TLabel", justify="left")
        self.scale_note.grid(row=4, column=0, columnspan=3, sticky="w", pady=(6, 0))

        save_row = ttk.Frame(card, style="Card.TFrame")
        save_row.grid(row=5, column=0, columnspan=3, sticky="ew", pady=(10, 0))
        self.where = ttk.Label(save_row, text="", style="CardSub.TLabel", justify="left")
        self.where.pack(side="left")
        self.save_btn = ttk.Button(save_row, text="Save to camera_config.yaml", command=self._save)
        self.save_btn.pack(side="right")

        # --- backends ------------------------------------------------------
        backends = ttk.LabelFrame(
            win, text=" Camera types ", style="Card.TLabelframe", padding=(14, 8)
        )
        backends.pack(fill="x", padx=16, pady=(12, 0))
        for i, b in enumerate(self.manager.backend_status()):
            ok = b["available"]
            ttk.Label(backends, text=b["title"], style="Card.TLabel").grid(
                row=i, column=0, sticky="w", pady=1
            )
            ttk.Label(
                backends, text="ready" if ok else "not available",
                style="CardGood.TLabel" if ok else "CardWarn.TLabel",
            ).grid(row=i, column=1, sticky="w", padx=(12, 0))
            why = b["detail"] if ok else b["detail"] + (
                f" - {b['install_hint']}" if b["install_hint"] else "")
            ttk.Label(backends, text=why, style="CardSub.TLabel", wraplength=440).grid(
                row=i, column=2, sticky="w", padx=(12, 0)
            )

        footer = ttk.Frame(win, padding=16)
        footer.pack(fill="x")
        ttk.Label(
            footer, text="Shortcuts in the viewer:  k  this window   t  next camera",
            style="Sub.TLabel",
        ).pack(side="left")
        ttk.Button(footer, text="Close", command=self.close).pack(side="right")

        self.visible = True
        self._refresh()

    # ------------------------------------------------------------------ #
    def _refresh(self):
        if self.win is None:
            return
        try:
            rows = self.manager.snapshot()
        except Exception:
            rows = []
        self._rows = {r["uid"]: r for r in rows}
        existing = set(self.tree.get_children())
        for index, r in enumerate(rows):
            values = (
                r["state_text"],
                r["label"],
                r["model"],
                r["serial"],
                r["backend_title"],
            )
            if r["uid"] in existing:
                self.tree.item(r["uid"], values=values, tags=(r["state"],))
                if self.tree.index(r["uid"]) != index:
                    self.tree.move(r["uid"], "", index)
            else:
                self.tree.insert("", index, iid=r["uid"], values=values, tags=(r["state"],))
        for gone in existing - set(self._rows):
            self.tree.delete(gone)
        if not self.tree.selection() and rows:
            active = next((r["uid"] for r in rows if r["active"]), rows[0]["uid"])
            self.tree.selection_set(active)
            self.tree.focus(active)
        self._show_details()
        if self.win.tk.call("clock", "milliseconds") > self._status_until:
            connecting = next((r for r in rows if r["state"] == "connecting"), None)
            self.status.configure(
                text=f"Connecting to {connecting['label']}..." if connecting else "",
                foreground=tkhost.SUBTEXT,
            )
        self.win.after(REFRESH_MS, self._refresh)

    def _selected(self) -> dict | None:
        sel = self.tree.selection()
        return self._rows.get(sel[0]) if sel else None

    def _on_select(self):
        self._form_uid = None  # refill the form for the new selection
        self._show_details()

    def _show_details(self):
        r = self._selected()
        if r is None:
            self.title.configure(text="No camera selected")
            self.detail.configure(text="")
            return
        self.title.configure(text=f"{r['label']}   ({r['state_text']})")
        parts = [r["model"]]
        if r["serial"]:
            parts.append(f"serial {r['serial']}")
        parts.append(r["backend_title"])
        lines = ["  ·  ".join(parts)]
        if r["sensor"]:
            lines.append(f"Sensor {r['sensor'][0]} x {r['sensor'][1]} px")
        if r["state"] == "lost":
            lines.append("Disconnected - it reconnects by itself when it comes back.")
        if r["error"] and r["state"] in ("error", "lost"):
            lines.append(f"Last error: {r['error']}")
        self.detail.configure(text="\n".join(lines))

        if r["active"]:
            self.use_btn.configure(text="In use", state="disabled")
        elif r["state"] == "absent":
            self.use_btn.configure(text="Not connected", state="disabled")
        else:
            self.use_btn.configure(text="Switch to this camera", state="normal")

        if self._form_uid != r["uid"]:  # never overwrite what the user is typing
            self._form_uid = r["uid"]
            self.label_var.set(r["configured_label"])
            px = r["pixel_size"]
            self.pixel_var.set(f"{px * 1e6:g}" if px else "")
        key = f"serial {r['serial']}" if r["serial"] else f"model {r['model']}"
        self.where.configure(
            text=f"Pixel size is saved for every {r['model']}; the label for {key}."
        )
        if r["pixel_size_known"]:
            self.scale_note.configure(text="")
        else:
            self.scale_note.configure(
                text="Pixel size not set: micron readings equal pixel counts until you "
                     "enter it."
            )

    # ------------------------------------------------------------------ #
    def _flash(self, text: str, error: bool = False):
        self.status.configure(text=text, foreground=BAD if error else tkhost.SUBTEXT)
        self._status_until = self.win.tk.call("clock", "milliseconds") + 6000

    def _use(self):
        r = self._selected()
        if r is None or r["active"] or r["state"] == "absent":
            return
        self.manager.post(("switch", r["uid"]))
        self._flash(f"Switching to {r['label']}...")

    def _rescan(self):
        self.manager.post(("rescan",))
        self._flash("Scanning for cameras...")

    def _save(self):
        r = self._selected()
        if r is None:
            return
        text = self.pixel_var.get().strip().replace(",", ".")
        try:
            pixel_size = float(f"{text}e-6") if text else None  # exact decimal, no 2.4999e-6
        except ValueError:
            self._flash("Pixel size must be a number, in micrometres", error=True)
            return
        if pixel_size is not None and not 0 < pixel_size < 1e-3:
            self._flash("Pixel size should be in micrometres, e.g. 2.74", error=True)
            return
        label = self.label_var.get().strip()
        path = cameraconfig.writable_path()
        changed = []
        try:
            if pixel_size != r["pixel_size"]:
                cameraconfig.save_entry(*self._pixel_size_home(r, path),
                                        {"pixel_size": pixel_size}, path)
                changed.append("pixel size")
            if label != r["configured_label"]:
                section, key = ("serials", r["serial"]) if r["serial"] else ("cameras", r["model"])
                cameraconfig.save_entry(section, key, {"label": label}, path)
                changed.append("label")
        except Exception as e:  # noqa: BLE001 - whatever it is, say so in the window
            if changed:  # the first edit is on disk: apply it, and say what is not
                self.manager.post(("reload_config",))
                self._flash(f"Saved {changed[0]}, but not the rest: {e}", error=True)
            else:
                self._flash(f"Not saved: {e}", error=True)
            return
        if not changed:
            self._flash("Nothing to save")
            return
        # The form already shows what was saved; it is not refilled from the
        # list, which is stale until the main loop has reloaded the file.
        self.manager.post(("reload_config",))
        self._flash(f"Saved {' and '.join(changed)} to {os.path.basename(path)}")

    @staticmethod
    def _pixel_size_home(r: dict, path: str) -> tuple[str, str]:
        """Where a pixel size takes effect: under the model, unless this unit's
        serial entry already sets one (it would win over the model's)."""
        try:
            serials = cameraconfig.load(path).serials
        except Exception:
            serials = {}
        entry = serials.get(r["serial"]) if r["serial"] else None
        if isinstance(entry, dict) and "pixel_size" in entry:
            return "serials", r["serial"]
        return "cameras", r["model"]

    # ------------------------------------------------------------------ #
    def exists(self) -> bool:
        try:
            return self.win is not None and bool(self.win.winfo_exists())
        except tk.TclError:
            return False

    def show(self):
        self.win.deiconify()
        self.win.lift()
        self.visible = True

    def close(self):
        self.visible = False
        if self.win is not None:
            self.win.withdraw()

    def destroy(self):
        self.visible = False
        if self.win is not None:
            win, self.win = self.win, None
            try:
                win.destroy()
            except tk.TclError:
                pass
