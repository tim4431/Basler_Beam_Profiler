"""Live settings window for the spot fit (press `o` in the viewer).

Runs its own Tk event loop on a daemon thread so the OpenCV grab/draw loop on
the main thread is never blocked.  Nothing is mutated in place: edits build a
new `FitConfig` and hand it to `fitconfig.set_active`, which swaps it in
atomically, so a frame in flight can never see a half-applied change.

Every control is generated from `fitconfig.SETTINGS`, so adding a knob there
adds it here with its range, its type and its explanation - there is no
per-field UI code to keep in sync.

The root is created once and merely hidden on close: Tk does not reliably
allow a second `tk.Tk()` in the same process after the first is destroyed
(it fails with "tk wasn't installed properly"), which would stop the window
reopening.  The daemon thread therefore lives for the life of the process.
"""

from __future__ import annotations

import logging
import threading
import tkinter as tk
from tkinter import ttk

from .. import fitconfig

SLIDER_SPAN = 200.0  # ranges wider than this get a plain entry box instead

# palette shared by every widget style (light, flat, single accent)
BG = "#eceff4"
CARD = "#ffffff"
BORDER = "#d8dee9"
ACCENT = "#2563eb"
ACCENT_DARK = "#1d4ed8"
TEXT = "#1f2430"
SUBTEXT = "#5b6472"
FIELD = "#f6f8fb"

_window: SettingsWindow | None = None
_lock = threading.Lock()


def open_settings():
    """Open the settings window, or raise it if it is already open."""
    global _window
    with _lock:
        if _window is not None:
            _window.show()
            return
        _window = SettingsWindow()
        threading.Thread(target=_window.run, name="fit-settings", daemon=True).start()


def close_settings():
    """Hide the window; the Tk root stays alive so it can be reopened."""
    with _lock:
        if _window is not None:
            _window.close()


def is_open() -> bool:
    """Whether the window is currently shown (read from any thread)."""
    w = _window
    return bool(w is not None and w.visible)


class SettingsWindow:
    def __init__(self):
        self.alive = False
        self.visible = False
        self._root: tk.Tk | None = None
        self._vars: dict[str, tk.Variable] = {}
        self._applying = False  # suppress the write-back while we refresh widgets
        self._shown = fitconfig.active()

    # ------------------------------------------------------------------ #
    def run(self):
        try:
            self._build()
            self.alive = True
            self._root.mainloop()
        except Exception as e:
            logging.error(f"settings window failed: {e}")
        finally:
            self.alive = False
            self.visible = False

    def _build(self):
        # widget construction fires the slider callbacks; ignore them until
        # every control exists, or _collect sees a half-built _vars
        self._applying = True
        self._root = root = tk.Tk()
        root.title("Spot fitting settings")
        root.minsize(580, 680)
        root.configure(background=BG)
        root.protocol("WM_DELETE_WINDOW", self.close)  # hide, keep the root
        self._style(root)

        header = ttk.Frame(root, padding=(16, 14, 16, 4))
        header.pack(fill="x")
        ttk.Label(header, text="Spot fitting", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            header,
            text="Changes apply to the next frame and are saved to fit_config.json.",
            style="Sub.TLabel",
        ).pack(anchor="w", pady=(2, 0))

        body = ttk.Frame(root, padding=(16, 0))
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
            root, height=7, wrap="word", relief="flat", borderwidth=0,
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

        buttons = ttk.Frame(root, padding=16)
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
        root.after(500, self._poll_external)

    @staticmethod
    def _style(root: tk.Tk):
        """Flat, light theme with one accent colour, on top of clam."""
        style = ttk.Style(root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        from tkinter import font as tkfont  # noqa: PLC0415  (needs a root first)

        base = tkfont.nametofont("TkDefaultFont")
        base.configure(size=10)
        family = base.actual("family")
        style.configure(".", background=BG, foreground=TEXT)
        style.configure("TFrame", background=BG)
        style.configure("TLabel", background=BG, foreground=TEXT)
        style.configure("Title.TLabel", font=(family, 14, "bold"))
        style.configure("Sub.TLabel", foreground=SUBTEXT)
        style.configure("Card.TLabel", background=CARD)
        style.configure(
            "Card.TLabelframe", background=CARD, relief="solid",
            borderwidth=1, bordercolor=BORDER,
        )
        style.configure(
            "Card.TLabelframe.Label", background=BG, foreground=ACCENT,
            font=(family, 10, "bold"),
        )
        style.configure("Card.TFrame", background=CARD)
        style.configure("Card.TCheckbutton", background=CARD)
        style.map("Card.TCheckbutton", background=[("active", CARD)])
        style.configure(
            "TEntry", fieldbackground=FIELD, bordercolor=BORDER,
            lightcolor=BORDER, darkcolor=BORDER, padding=3,
        )
        style.configure(
            "Horizontal.TScale", troughcolor=BORDER, background=CARD,
            lightcolor=CARD, darkcolor=CARD,
        )
        style.configure(
            "TButton", padding=(14, 7), background="#e2e7ef", bordercolor=BORDER,
            lightcolor="#e2e7ef", darkcolor="#e2e7ef", focusthickness=0,
        )
        style.map("TButton", background=[("active", "#d3dae6"), ("pressed", "#c6cedd")])
        style.configure(
            "Accent.TButton", background=ACCENT, foreground="#ffffff",
            bordercolor=ACCENT, lightcolor=ACCENT, darkcolor=ACCENT,
        )
        style.map(
            "Accent.TButton",
            background=[("active", ACCENT_DARK), ("pressed", ACCENT_DARK)],
        )

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
        if not self.alive and self._root is None:
            return
        cfg = fitconfig.active()
        if cfg != self._shown:
            self._show(cfg)
        self._root.after(500, self._poll_external)

    def _set_help(self, text: str):
        self._help.configure(state="normal")
        self._help.delete("1.0", "end")
        self._help.insert("1.0", text)
        self._help.configure(state="disabled")

    def show(self):
        if self._root is not None:
            self._root.after(0, self._on_show)

    def _on_show(self):
        self._show(fitconfig.active())
        self._root.deiconify()
        self._root.lift()
        self.visible = True

    def close(self):
        """Hide, do not destroy - see the module docstring."""
        self.visible = False
        if self._root is not None:
            try:
                self._root.after(0, self._root.withdraw)
            except tk.TclError:
                pass

    def destroy(self):
        """Tear the root down for good.  Only used by the tests."""
        self.alive = False
        self.visible = False
        if self._root is not None:
            root, self._root = self._root, None
            try:
                root.destroy()
            except tk.TclError:
                pass
