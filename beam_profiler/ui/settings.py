"""Live settings window for the spot fit (press `o` in the viewer).

A Toplevel on the shared Tk thread (see `tkhost`), so the OpenCV grab/draw
loop on the main thread is never blocked.  Nothing is mutated in place: edits
build a new `FitConfig` and hand it to `fitconfig.set_active`, which swaps it
in atomically, so a frame in flight can never see a half-applied change.

Every control is generated from `fitconfig.SETTINGS`, so adding a knob there
adds it here with its range, its type and its explanation - there is no
per-field UI code to keep in sync.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from .. import fitconfig
from . import tkhost
from .tkhost import BORDER, FIELD, SUBTEXT

SLIDER_SPAN = 200.0  # ranges wider than this get a plain entry box instead

_window: SettingsWindow | None = None


def open_settings():
    """Open the settings window, or raise it if it is already open."""
    tkhost.call(_open)


def _open(root):
    global _window
    if _window is None or not _window.exists():
        _window = SettingsWindow()
        _window.build(root)
    else:
        _window.show()


def close_settings():
    """Hide the window (it is kept, so reopening is instant)."""
    tkhost.call(lambda _root: _window.close() if _window is not None else None)


def is_open() -> bool:
    """Whether the window is currently shown (read from any thread)."""
    w = _window
    return bool(w is not None and w.visible)


class SettingsWindow:
    def __init__(self):
        self.visible = False
        self.win: tk.Toplevel | None = None
        self._vars: dict[str, tk.Variable] = {}
        self._applying = False  # suppress the write-back while we refresh widgets
        self._shown = fitconfig.active()

    # ------------------------------------------------------------------ #
    def build(self, master: tk.Misc):
        """Create the window as a Toplevel of `master`, on the Tk thread."""
        # widget construction fires the slider callbacks; ignore them until
        # every control exists, or _collect sees a half-built _vars
        self._applying = True
        self.win = win = tk.Toplevel(master)
        win.title("Spot fitting settings")
        win.minsize(580, 680)
        win.configure(background=tkhost.BG)
        win.protocol("WM_DELETE_WINDOW", self.close)

        header = ttk.Frame(win, padding=(16, 14, 16, 4))
        header.pack(fill="x")
        ttk.Label(header, text="Spot fitting", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            header,
            text="Changes apply to the next frame and are saved to fit_config.json.",
            style="Sub.TLabel",
        ).pack(anchor="w", pady=(2, 0))

        body = ttk.Frame(win, padding=(16, 0))
        body.pack(fill="both", expand=True)

        cfg = fitconfig.active()
        groups: dict[str, ttk.LabelFrame] = {}
        for s in fitconfig.SETTINGS:
            if s.group not in groups:
                frame = ttk.LabelFrame(
                    body, text=f" {s.group} ", style="Card.TLabelframe", padding=(12, 8)
                )
                frame.pack(fill="x", pady=(8, 0))
                frame.columnconfigure(1, weight=1)
                groups[s.group] = frame
            self._add_row(groups[s.group], s, getattr(cfg, s.name))

        self._help = tk.Text(
            win, height=7, wrap="word", relief="flat", borderwidth=0,
            background=FIELD, foreground=SUBTEXT, padx=12, pady=10,
            highlightthickness=1, highlightbackground=BORDER, highlightcolor=BORDER,
        )
        self._help.pack(fill="both", expand=False, padx=16, pady=(12, 0))
        self._help.configure(state="disabled")
        self._set_help(
            "Hover or focus a setting to see what it does.\n\n"
            "Defaults are the measured optima. docs/fitting.md explains the method; "
            "docs/fitting-calibration.md shows the measurements behind each default."
        )

        buttons = ttk.Frame(win, padding=16)
        buttons.pack(fill="x")
        ttk.Button(buttons, text="Restore defaults", command=self._defaults).pack(side="left")
        ttk.Button(buttons, text="Reload from file", command=self._reload).pack(
            side="left", padx=(8, 0)
        )
        ttk.Button(buttons, text="Close", style="Accent.TButton", command=self.close).pack(
            side="right"
        )

        self._applying = False
        self.visible = True
        win.after(500, self._poll_external)

    def _add_row(self, parent: ttk.LabelFrame, s: fitconfig.Setting, value):
        row = parent.grid_size()[1]
        label = ttk.Label(parent, text=s.label, style="Card.TLabel")
        label.grid(row=row, column=0, sticky="w", pady=3)

        if s.kind == "bool":
            var = tk.BooleanVar(value=bool(value))
            widget = ttk.Checkbutton(
                parent, variable=var, command=self._apply, style="Card.TCheckbutton"
            )
            widget.grid(row=row, column=1, sticky="w", padx=(10, 0))
        else:
            is_int = s.kind == "int"
            var = tk.StringVar(value=str(int(value) if is_int else round(float(value), 4)))
            holder = ttk.Frame(parent, style="Card.TFrame")
            holder.grid(row=row, column=1, sticky="ew", padx=(10, 0))
            holder.columnconfigure(0, weight=1)
            if s.hi - s.lo <= SLIDER_SPAN:
                scale = ttk.Scale(
                    holder, from_=s.lo, to=s.hi,
                    command=lambda v, st=s, sv=var: self._on_slide(st, sv, v),
                )
                scale.set(float(value))
                scale.grid(row=0, column=0, sticky="ew")
                self._vars[s.name + ":scale"] = scale
            entry = ttk.Entry(holder, textvariable=var, width=9, justify="right")
            entry.grid(row=0, column=1, padx=(8, 0))
            entry.bind("<Return>", lambda _e: self._apply())
            entry.bind("<FocusOut>", lambda _e: self._apply())
            widget = entry

        self._vars[s.name] = var
        for w in (label, widget):
            w.bind("<Enter>", lambda _e, st=s: self._set_help(f"{st.label}\n\n{st.help}"))
            w.bind("<FocusIn>", lambda _e, st=s: self._set_help(f"{st.label}\n\n{st.help}"))

    # ------------------------------------------------------------------ #
    def _on_slide(self, s: fitconfig.Setting, var: tk.StringVar, raw: str):
        if self._applying:
            return
        value = float(raw)
        var.set(str(int(round(value)) if s.kind == "int" else round(value, 4)))
        self._apply()

    def _collect(self) -> fitconfig.FitConfig:
        """Read every widget; anything unparseable keeps the value in force."""
        current = fitconfig.active()
        values = {}
        for s in fitconfig.SETTINGS:
            var = self._vars.get(s.name)
            try:
                values[s.name] = bool(var.get()) if s.kind == "bool" else float(var.get())
            except (AttributeError, ValueError, tk.TclError):
                values[s.name] = getattr(current, s.name)
        return fitconfig.clamp(fitconfig.FitConfig(**values))

    def _apply(self):
        if self._applying:
            return
        cfg = self._collect()
        if cfg != fitconfig.active():
            fitconfig.set_active(cfg)
        self._show(cfg)

    def _show(self, cfg: fitconfig.FitConfig):
        """Push a config into the widgets without triggering a write-back."""
        self._applying = True
        try:
            for s in fitconfig.SETTINGS:
                value = getattr(cfg, s.name)
                var = self._vars[s.name]
                if s.kind == "bool":
                    var.set(bool(value))
                else:
                    var.set(str(int(value) if s.kind == "int" else round(float(value), 4)))
                scale = self._vars.get(s.name + ":scale")
                if scale is not None:
                    scale.set(float(value))
            self._shown = cfg
        finally:
            self._applying = False

    def _defaults(self):
        fitconfig.set_active(fitconfig.FitConfig())
        self._show(fitconfig.active())

    def _reload(self):
        fitconfig.set_active(fitconfig.load_config())
        self._show(fitconfig.active())

    def _poll_external(self):
        """Pick up changes made elsewhere (HTTP API, another window)."""
        if self.win is None:
            return
        cfg = fitconfig.active()
        if cfg != self._shown:
            self._show(cfg)
        self.win.after(500, self._poll_external)

    def _set_help(self, text: str):
        self._help.configure(state="normal")
        self._help.delete("1.0", "end")
        self._help.insert("1.0", text)
        self._help.configure(state="disabled")

    def exists(self) -> bool:
        try:
            return self.win is not None and bool(self.win.winfo_exists())
        except tk.TclError:
            return False

    def show(self):
        """Raise the window with current values (on the Tk thread)."""
        self._show(fitconfig.active())
        self.win.deiconify()
        self.win.lift()
        self.visible = True

    def close(self):
        """Hide, keeping the widgets for a quick reopen (on the Tk thread)."""
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
