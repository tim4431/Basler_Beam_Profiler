"""Fitting controls serviced on the main GUI thread, including on macOS."""

from __future__ import annotations

import logging
import tkinter as tk
from tkinter import ttk

from .. import fitconfig

SLIDER_SPAN = 200.0  # ranges wider than this get a plain entry box instead

_window: SettingsWindow | None = None
_gui_root: tk.Tk | None = None


def prepare_gui():
    """Tk must initialize Cocoa before OpenCV creates NSApplication on macOS."""
    global _gui_root
    if _gui_root is None:
        _gui_root = tk.Tk()
        _gui_root.withdraw()
    return _gui_root


def open_settings():
    """Create once on the main thread; the viewer pumps Tk between frames."""
    global _window
    if _window is None:
        window = SettingsWindow()
        try:
            window._build()
            window.alive = True
        except Exception:
            window.destroy()
            raise
        _window = window
    else:
        _window.show()


def pump_events():
    if _gui_root is not None:
        _gui_root.update_idletasks()
        _gui_root.update()


def close_settings():
    if _window is not None:
        _window.close()


def destroy_settings():
    global _window, _gui_root
    if _window is not None:
        _window.destroy()
        _window = None
    if _gui_root is not None:
        _gui_root.destroy()
        _gui_root = None


def save_dialog(**options):
    from tkinter import filedialog
    # Reuse the existing Tk interpreter; do not create/destroy a second root.
    return filedialog.asksaveasfilename(parent=prepare_gui(), **options)


class SettingsWindow:
    def __init__(self):
        self.alive = False
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

    def _build(self):
        # widget construction fires the slider callbacks; ignore them until
        # every control exists, or _collect sees a half-built _vars
        self._applying = True
        self._root = root = tk.Toplevel(prepare_gui())
        root.title("Spot fitting settings")
        root.minsize(560, 640)
        root.protocol("WM_DELETE_WINDOW", self.close)  # hide, keep the root

        header = ttk.Label(
            root,
            text="Changes apply to the next frame and are saved to fit_config.json.",
            padding=(10, 8),
        )
        header.pack(fill="x")

        body = ttk.Frame(root, padding=(10, 0))
        body.pack(fill="both", expand=True)

        cfg = fitconfig.active()
        groups: dict[str, ttk.LabelFrame] = {}
        for s in fitconfig.SETTINGS:
            if s.group not in groups:
                frame = ttk.LabelFrame(body, text=s.group, padding=(10, 6))
                frame.pack(fill="x", pady=(6, 0))
                frame.columnconfigure(1, weight=1)
                groups[s.group] = frame
            self._add_row(groups[s.group], s, getattr(cfg, s.name))

        self._help = tk.Text(root, height=8, wrap="word", relief="sunken", borderwidth=1)
        self._help.pack(fill="both", expand=False, padx=10, pady=(10, 0))
        self._help.configure(state="disabled")
        self._set_help(
            "Hover or focus a setting to see what it does.\n\n"
            "Defaults are the measured optima. docs/fitting.md explains the method; "
            "docs/fitting-calibration.md shows the measurements behind each default."
        )

        buttons = ttk.Frame(root, padding=10)
        buttons.pack(fill="x")
        ttk.Button(buttons, text="Restore defaults", command=self._defaults).pack(side="left")
        ttk.Button(buttons, text="Reload from file", command=self._reload).pack(
            side="left", padx=(8, 0)
        )
        ttk.Button(buttons, text="Close", command=self.close).pack(side="right")

        self._applying = False
        root.after(500, self._poll_external)

    def _add_row(self, parent: ttk.LabelFrame, s: fitconfig.Setting, value):
        row = parent.grid_size()[1]
        label = ttk.Label(parent, text=s.label)
        label.grid(row=row, column=0, sticky="w", pady=3)

        if s.kind == "bool":
            var = tk.BooleanVar(value=bool(value))
            widget = ttk.Checkbutton(parent, variable=var, command=self._apply)
            widget.grid(row=row, column=1, sticky="w", padx=(10, 0))
        else:
            is_int = s.kind == "int"
            var = tk.StringVar(value=str(int(value) if is_int else round(float(value), 4)))
            holder = ttk.Frame(parent)
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

    def close(self):
        """Hide, do not destroy - see the module docstring."""
        if self._root is not None:
            try:
                self._root.after(0, self._root.withdraw)
            except tk.TclError:
                pass

    def destroy(self):
        """Tear the root down for good.  Only used by the tests."""
        self.alive = False
        if self._root is not None:
            root, self._root = self._root, None
            try:
                root.destroy()
            except tk.TclError:
                pass
