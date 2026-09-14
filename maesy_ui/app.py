"""Tkinter frontend generated from registered argparse command trees."""

from __future__ import annotations

import queue
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

try:
    import tkinter as tk
    from tkinter import messagebox, ttk
except ImportError as exc:  # pragma: no cover - platform dependent
    raise RuntimeError("MaeSy UI requires Python's Tkinter module.") from exc

try:
    from PIL import Image, ImageTk
except ImportError:  # pragma: no cover - optional dependency guard
    Image = None
    ImageTk = None

from .argparse_adapter import (
    ArgumentField,
    ArgumentValidationError,
    build_argv,
    command_choices,
    default_value,
    fields_for_path,
    validate_argv,
)
from .registry import ArchitectureTree, CliTarget, registered_targets


def browser_selection(
    current_folder: Path, highlighted: Path | None, highlighted_is_file: bool
) -> Path:
    """Resolve a browser Select action without depending on Tk widgets.

    If a file or folder is explicitly highlighted, use it. Otherwise, select the
    currently open folder. This mirrors the requested browser semantics.
    """
    if highlighted is not None:
        return highlighted if highlighted_is_file or highlighted.is_dir() else current_folder
    return current_folder


@dataclass
class _WidgetField:
    field: ArgumentField
    widget: tk.Widget
    get_value: Callable[[], Any]
    set_value: Callable[[str], None]

    def value(self) -> Any:
        return self.get_value()


CUSTOM_PATH = "Custom path…"


class _Tooltip:
    def __init__(self, widget: tk.Widget, text: str):
        self.widget = widget
        self.text = text
        self.popup: tk.Toplevel | None = None
        widget.bind("<Enter>", self.show, add="+")
        widget.bind("<Leave>", self.hide, add="+")
        widget.bind("<FocusIn>", self.show, add="+")
        widget.bind("<FocusOut>", self.hide, add="+")

    def show(self, _event: object = None) -> None:
        if self.popup or not self.text:
            return
        self.popup = tk.Toplevel(self.widget)
        self.popup.wm_overrideredirect(True)
        self.popup.geometry(f"+{self.widget.winfo_rootx() + 14}+{self.widget.winfo_rooty() + self.widget.winfo_height() + 6}")
        tk.Label(self.popup, text=self.text, justify="left", wraplength=440, background="#1b2722", foreground="#f0f7df", padx=8, pady=6).pack()

    def hide(self, _event: object = None) -> None:
        if self.popup:
            self.popup.destroy()
            self.popup = None


class MaesyUiApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.targets = registered_targets()
        if not self.targets:
            raise RuntimeError("No CLI targets are registered for MaeSy UI.")

        self.target_by_name = {target.display_name: target for target in self.targets}
        self.target_var = tk.StringVar(value=self.targets[0].display_name)
        self.subcommand_values: dict[tuple[str, ...], str] = {}
        self.widget_fields: dict[tuple[tuple[str, ...], str], _WidgetField] = {}
        self.form_values: dict[tuple[tuple[str, ...], str], Any] = {}
        self.output_queue: queue.Queue[tuple[str, str | int | None]] = queue.Queue()
        self.running = False
        self.process: subprocess.Popen[str] | None = None
        self.status_var = tk.StringVar(value="Ready")
        self.mutable_controls: list[tk.Widget] = []

        self.root.title("MaeSy UI")
        self.root.minsize(860, 620)
        self.logo = self._load_logo()
        if self.logo is not None:
            try:
                self.root.iconphoto(True, self.logo)
            except tk.TclError:
                pass
        self._configure_style()
        self._build_layout()
        self._render_form()
        self.root.after(50, self._drain_output)

    def _load_logo(self) -> tk.PhotoImage | None:
        logo_path = Path(__file__).with_name("Images") / "ruhrbotdevils_logo.png"
        if not logo_path.exists():
            return None
        try:
            if Image is not None and ImageTk is not None:
                with Image.open(logo_path) as image:
                    image = image.copy().convert("RGBA")
                    image.thumbnail((48, 48), Image.Resampling.LANCZOS)
                    return ImageTk.PhotoImage(image)
            return tk.PhotoImage(file=str(logo_path))
        except Exception:
            try:
                return tk.PhotoImage(file=str(logo_path))
            except Exception:
                return None

    @property
    def target(self) -> CliTarget:
        return self.target_by_name[self.target_var.get()]

    def _configure_style(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        black, charcoal, panel, green, mint = "#090d0c", "#121a17", "#1b2722", "#639A00", "#f0f7df"
        self.root.configure(background=black)
        style.configure("TFrame", background=charcoal)
        style.configure("TLabelframe", background=panel, bordercolor="#2c4037", relief="flat")
        style.configure("TLabelframe.Label", background=panel, foreground=mint, font=("TkDefaultFont", 10, "bold"))
        style.configure("TLabel", background=charcoal, foreground=mint)
        style.configure("TLabelframe.TLabel", background=panel, foreground=mint)
        style.configure("TEntry", fieldbackground="#0f1714", foreground=mint, insertcolor=mint, padding=6, bordercolor=green, lightcolor=green, darkcolor=green)
        style.map("TEntry", bordercolor=[("focus", green)], lightcolor=[("focus", green)], darkcolor=[("focus", green)])
        style.configure("TButton", padding=(11, 6), background="#26362f", foreground=mint, borderwidth=0, bordercolor=green, lightcolor=green, darkcolor=green)
        style.map("TButton", background=[("active", "#33483e"), ("disabled", "#1b2722")], bordercolor=[("focus", green)], lightcolor=[("focus", green)], darkcolor=[("focus", green)])
        style.configure("Accent.TButton", background=green, foreground=black, font=("TkDefaultFont", 10, "bold"))
        style.map("Accent.TButton", background=[("active", "#7cad1f"), ("disabled", "#3b5420")], foreground=[("disabled", "#a9bd81")])
        style.configure("TCombobox", padding=5, fieldbackground="#0f1714", background="#26362f", foreground=mint, arrowcolor=green, bordercolor=green, lightcolor=green, darkcolor=green)
        style.map("TCombobox", fieldbackground=[("readonly", "#0f1714")], foreground=[("readonly", mint)], selectbackground=[("readonly", green)], selectforeground=[("readonly", black)], bordercolor=[("focus", green)], lightcolor=[("focus", green)], darkcolor=[("focus", green)])
        style.configure("Green.Horizontal.TProgressbar", troughcolor="#23352d", background=green, lightcolor=green, darkcolor=green, bordercolor="#23352d")
        self.root.option_add("*Dialog.msg.font", ("TkDefaultFont", 10))
        self.root.option_add("*TCombobox*Listbox*Background", "#1b2722")
        self.root.option_add("*TCombobox*Listbox*Foreground", mint)
        self.root.option_add("*TCombobox*Listbox*selectBackground", green)
        self.root.option_add("*TCombobox*Listbox*selectForeground", black)
        self.root.option_add("*Text.background", "#0f1714")
        self.root.option_add("*Text.foreground", mint)
        self.root.option_add("*Text.insertBackground", mint)
        self.root.option_add("*Text.selectBackground", green)
        self.root.option_add("*Text.selectForeground", black)
        self.root.option_add("*Listbox.background", "#0f1714")
        self.root.option_add("*Listbox.foreground", mint)
        self.root.option_add("*Listbox.selectBackground", green)
        self.root.option_add("*Listbox.selectForeground", black)
        style.configure("Vertical.TScrollbar", background="#26362f", troughcolor="#0f1714", bordercolor="#26362f", arrowcolor=green)
        self.root.bind_class("FormScrollable", "<MouseWheel>", self._scroll_form)
        self.root.bind_class("FormScrollable", "<Button-4>", self._scroll_form)
        self.root.bind_class("FormScrollable", "<Button-5>", self._scroll_form)
        self.root.bind_class("TCombobox", "<MouseWheel>", self._scroll_form)
        self.root.bind_class("TCombobox", "<Button-4>", self._scroll_form)
        self.root.bind_class("TCombobox", "<Button-5>", self._scroll_form)

    def _register_mutable(self, widget: tk.Widget) -> tk.Widget:
        self.mutable_controls.append(widget)
        return widget

    def _update_form_scrollregion(self, _event: object = None) -> None:
        self.form_canvas.configure(scrollregion=self.form_canvas.bbox("all"))

    def _resize_form_window(self, event: tk.Event) -> None:
        self.form_canvas.itemconfigure(self.form_window, width=event.width)

    def _scroll_form(self, event: tk.Event | None = None) -> str | None:
        if event is None:
            return None
        if hasattr(event, "delta") and event.delta:
            self.form_canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")
        elif hasattr(event, "num"):
            if event.num == 4:
                self.form_canvas.yview_scroll(-1, "units")
            elif event.num == 5:
                self.form_canvas.yview_scroll(1, "units")
        return "break"

    def _mark_form_scrollable(self, widget: tk.Widget) -> None:
        tags = widget.bindtags()
        if "FormScrollable" not in tags:
            widget.bindtags((tags[0], "FormScrollable", *tags[1:]))
        for child in widget.winfo_children():
            self._mark_form_scrollable(child)

    def _set_running_state(self, running: bool) -> None:
        """Lock command inputs while keeping the output pane responsive."""
        state = "disabled" if running else "normal"
        for widget in tuple(self.mutable_controls):
            if not widget.winfo_exists():
                continue
            try:
                if isinstance(widget, ttk.Combobox) and not running:
                    widget.configure(state="readonly")
                else:
                    widget.configure(state=state)
            except tk.TclError:
                pass
        self.run_button.configure(state=state)
        if running:
            self._animate_progress()
        else:
            self.progress.stop()

    def _animate_progress(self) -> None:
        if not self.running:
            return
        self.progress.step(4)
        self.root.after(45, self._animate_progress)

    def _build_layout(self) -> None:
        outer = ttk.Frame(self.root, padding=12)
        outer.grid(sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(1, weight=1)
        outer.rowconfigure(3, weight=1)

        target_row = ttk.Frame(outer, padding=(4, 0, 4, 4))
        target_row.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        target_row.columnconfigure(2, weight=1)
        if self.logo is not None:
            ttk.Label(target_row, image=self.logo).grid(row=0, column=0, sticky="w", padx=(0, 12))
        selector = ttk.Combobox(
            target_row,
            textvariable=self.target_var,
            state="readonly",
            values=[target.display_name for target in self.targets],
            width=22,
        )
        selector.set(self.targets[0].display_name)
        selector.grid(row=0, column=1, sticky="w")
        self._register_mutable(selector)
        selector.bind("<<ComboboxSelected>>", self._on_target_changed)

        form_frame = ttk.LabelFrame(outer, text="Command arguments", padding=0)
        form_frame.grid(row=1, column=0, sticky="nsew")
        form_frame.columnconfigure(0, weight=1)
        form_frame.rowconfigure(0, weight=1)
        self.form_canvas = tk.Canvas(
            form_frame, highlightthickness=0, background="#1b2722", bd=0, relief="flat"
        )
        scrollbar = ttk.Scrollbar(form_frame, orient="vertical", command=self.form_canvas.yview)
        self.form_canvas.configure(yscrollcommand=scrollbar.set)
        self.form_canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.form = ttk.Frame(self.form_canvas, padding=12)
        self.form.columnconfigure(1, weight=1)
        self.form_window = self.form_canvas.create_window((0, 0), window=self.form, anchor="nw")
        self.form.bind("<Configure>", self._update_form_scrollregion)
        self.form_canvas.bind("<Configure>", self._resize_form_window)
        self.form_canvas.bind("<MouseWheel>", self._scroll_form)
        self.form_canvas.bind("<Button-4>", self._scroll_form)
        self.form_canvas.bind("<Button-5>", self._scroll_form)

        controls = ttk.Frame(outer)
        controls.grid(row=2, column=0, sticky="ew", pady=10)
        controls.columnconfigure(3, weight=1)
        self.run_button = ttk.Button(controls, text="Run", style="Accent.TButton", command=self.run)
        self.run_button.grid(row=0, column=0, sticky="w")
        self.progress = ttk.Progressbar(controls, mode="indeterminate", length=150, style="Green.Horizontal.TProgressbar")
        self.progress.grid(row=0, column=1, sticky="w", padx=(12, 0))
        ttk.Button(controls, text="Clear output", command=self._clear_output).grid(
            row=0, column=2, sticky="w", padx=(8, 0)
        )
        ttk.Label(controls, textvariable=self.status_var).grid(
            row=0, column=3, sticky="e"
        )

        output_frame = ttk.LabelFrame(outer, text="Live output", padding=6)
        output_frame.grid(row=3, column=0, sticky="nsew")
        output_frame.columnconfigure(0, weight=1)
        output_frame.rowconfigure(0, weight=1)
        self.output = tk.Text(
            output_frame,
            height=12,
            wrap="word",
            state="disabled",
            background="#0b100e",
            foreground="#eaf8ef",
            insertbackground="#eaf8ef",
            selectbackground="#639A00",
            selectforeground="#090d0c",
            relief="flat",
            padx=8,
            pady=8,
        )
        self.output.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(output_frame, command=self.output.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.output.configure(yscrollcommand=scrollbar.set)

    def _on_target_changed(self, _event: object = None) -> None:
        self._capture_values()
        self.subcommand_values.clear()
        self.form_values.clear()
        self._render_form()

    def _selected_path(self) -> tuple[str, ...]:
        parser = self.target.parser()
        path: list[str] = []
        while choices := command_choices(parser):
            prefix = tuple(path)
            selection = self.subcommand_values.get(prefix)
            if selection not in choices:
                selection = next(iter(choices))
                self.subcommand_values[prefix] = selection
            path.append(selection)
            parser = choices[selection]
        return tuple(path)

    def _on_subcommand_changed(self, prefix: tuple[str, ...], variable: tk.StringVar) -> None:
        self._capture_values()
        self.subcommand_values[prefix] = variable.get()
        for key in tuple(self.subcommand_values):
            if len(key) > len(prefix) and key[: len(prefix)] == prefix:
                del self.subcommand_values[key]
        self._render_form()

    def _render_form(self) -> None:
        self.mutable_controls = [
            widget for widget in self.mutable_controls if widget.winfo_exists() and widget is not self.form
        ]
        for child in self.form.winfo_children():
            child.destroy()
        self.mutable_controls = [widget for widget in self.mutable_controls if widget.winfo_exists()]
        self.widget_fields.clear()

        parser = self.target.parser()
        path: list[str] = []
        row = 0
        while choices := command_choices(parser):
            prefix = tuple(path)
            selected = self.subcommand_values.get(prefix, next(iter(choices)))
            self.subcommand_values[prefix] = selected
            variable = tk.StringVar(value=selected)
            ttk.Label(self.form, text="Command" if not path else "Subcommand").grid(
                row=row, column=0, sticky="nw", padx=(0, 8), pady=4
            )
            selector = ttk.Combobox(
                self.form,
                textvariable=variable,
                values=list(choices),
                state="readonly",
            )
            selector.grid(row=row, column=1, sticky="ew", pady=4)
            self._register_mutable(selector)
            selector.bind(
                "<<ComboboxSelected>>",
                lambda _event, key=prefix, var=variable: self._on_subcommand_changed(key, var),
            )
            path.append(selected)
            parser = choices[selected]
            row += 1

        for field in fields_for_path(self.target.parser(), path):
            self._render_field(row, field)
            row += 1
        self._mark_form_scrollable(self.form)
        self.form_canvas.yview_moveto(0)
        self.root.after_idle(lambda: self.form_canvas.yview_moveto(0))

    def _render_field(self, row: int, field: ArgumentField) -> None:
        action = field.action
        label = action.dest.replace("_", " ").capitalize()
        if field.required:
            label += " *"
        ttk.Label(self.form, text=label).grid(
            row=row, column=0, sticky="nw", padx=(0, 10), pady=5
        )
        current = self.form_values.get(field.key, default_value(field))
        hint = self.target.field_hint(field)
        if hint and hint.kind == "model-selector":
            widget_field = self._render_model_selector(field, current, hint.choices_provider())
        else:
            widget_field = self._render_standard_field(field, current)
        widget_field.widget.grid(row=row, column=1, sticky="ew", pady=5)
        if field.is_path_like and not (hint and hint.kind == "model-selector"):
            browse = ttk.Button(
                self.form,
                text="Browse…",
                command=lambda item=widget_field: self._browse_for_path(item),
            )
            browse.grid(row=row, column=2, sticky="nw", padx=(8, 0), pady=5)
            self._register_mutable(browse)
        help_text = action.help or ""
        if field.is_list:
            help_text = f"{help_text} Enter one value per line.".strip()
        if help_text:
            _Tooltip(widget_field.widget, help_text)
            _Tooltip(self.form.grid_slaves(row=row, column=0)[0], help_text)
        if not (hint and hint.kind == "model-selector"):
            self._register_mutable(widget_field.widget)
        self.widget_fields[field.key] = widget_field

    def _create_toggle(self, parent: tk.Misc, variable: tk.BooleanVar) -> tk.Canvas:
        """Create a compact dark toggle square with a TU-green selected cross."""
        toggle = tk.Canvas(
            parent,
            width=20,
            height=20,
            background="#0f1714",
            highlightthickness=1,
            highlightbackground="#639A00",
            highlightcolor="#639A00",
            bd=0,
            cursor="hand2",
        )

        def redraw(*_args: object) -> None:
            toggle.delete("mark")
            if variable.get():
                toggle.create_line(5, 5, 15, 15, fill="#639A00", width=2, tags="mark")
                toggle.create_line(15, 5, 5, 15, fill="#639A00", width=2, tags="mark")

        def toggle_value(_event: tk.Event) -> None:
            variable.set(not variable.get())

        variable.trace_add("write", redraw)
        toggle.bind("<Button-1>", toggle_value)
        redraw()
        return toggle

    def _render_standard_field(self, field: ArgumentField, current: Any) -> _WidgetField:
        action = field.action
        if field.is_boolean:
            variable = tk.BooleanVar(value=bool(current))
            widget = self._create_toggle(self.form, variable)
            return _WidgetField(field, widget, variable.get, lambda value: variable.set(bool(value)))
        if field.fixed_length is not None:
            container = ttk.Frame(self.form)
            current_values = list(current or []) if not isinstance(current, str) else current.split()
            variables: list[tk.StringVar] = []
            for index, label in enumerate(field.item_labels):
                item = ttk.Frame(container)
                item.grid(row=0, column=index, padx=(0, 8), sticky="w")
                ttk.Label(item, text=label).pack(anchor="w")
                variable = tk.StringVar(value=str(current_values[index]) if index < len(current_values) else "")
                ttk.Entry(item, textvariable=variable, width=9).pack(anchor="w")
                variables.append(variable)
            return _WidgetField(field, container, lambda: [variable.get() for variable in variables], lambda _value: None)
        if field.is_list:
            widget = tk.Text(
                self.form,
                height=3,
                width=50,
                background="#0f1714",
                foreground="#f0f7df",
                insertbackground="#f0f7df",
                selectbackground="#639A00",
                selectforeground="#090d0c",
                highlightthickness=1,
                highlightbackground="#639A00",
                highlightcolor="#639A00",
                relief="flat",
                bd=0,
                padx=7,
                pady=6,
            )
            lines = current if isinstance(current, str) else "\n".join(map(str, current or []))
            widget.insert("1.0", lines)
            return _WidgetField(
                field,
                widget,
                lambda: widget.get("1.0", "end-1c"),
                lambda value: widget.insert("end", ("\n" if widget.get("1.0", "end-1c") else "") + value),
            )
        variable = tk.StringVar(value="" if current is None else str(current))
        if action.choices:
            widget = ttk.Combobox(
                self.form,
                textvariable=variable,
                values=[str(choice) for choice in action.choices],
                state="readonly",
            )
        else:
            widget = ttk.Entry(self.form, textvariable=variable, width=50)
        return _WidgetField(field, widget, variable.get, variable.set)

    def _render_model_selector(
        self, field: ArgumentField, current: Any, architectures: ArchitectureTree
    ) -> _WidgetField:
        container = ttk.Frame(self.form)
        container.columnconfigure(0, weight=1)
        selectors: list[tuple[tk.StringVar, ArchitectureTree, tuple[str, ...]]] = []
        custom_var = tk.StringVar()

        def clear_after(depth: int) -> None:
            for child in container.grid_slaves(row=0):
                if int(child.grid_info()["column"]) > depth:
                    child.destroy()
            del selectors[depth + 1 :]

        def render_level(node: ArchitectureTree, prefix: tuple[str, ...], selected: str | None = None) -> None:
            depth = len(selectors)
            variable = tk.StringVar(value=selected or "")
            values = [f"{name} ›" if child is not None else name for name, child in node.items()]
            if depth == 0:
                values.append(CUSTOM_PATH)
            combo = ttk.Combobox(container, textvariable=variable, values=values, state="readonly", width=18)
            combo.grid(row=0, column=depth, sticky="ew", padx=(0, 6))
            self._register_mutable(combo)
            selectors.append((variable, node, prefix))

            def changed(_event: object = None) -> None:
                clear_after(depth)
                choice = variable.get()
                if choice == CUSTOM_PATH:
                    custom_entry.grid(row=1, column=0, sticky="ew", pady=(5, 0))
                    browse.grid(row=1, column=1, sticky="w", padx=(8, 0), pady=(5, 0))
                    return
                custom_entry.grid_remove()
                browse.grid_remove()
                name = choice.removesuffix(" ›")
                child = node.get(name)
                if child is not None:
                    render_level(child, (*prefix, name))

            combo.bind("<<ComboboxSelected>>", changed)

        custom_entry = ttk.Entry(container, textvariable=custom_var)
        browse = ttk.Button(container, text="Browse…", command=lambda: self._browse_model_path(custom_var))
        self._register_mutable(custom_entry)
        self._register_mutable(browse)
        canonical = str(current or "")
        parts = tuple(part for part in canonical.split("/") if part)
        node = architectures
        valid = bool(parts)
        for part in parts:
            child = node.get(part)
            if child is None and part not in node:
                valid = False
                break
            node = child or {}
        if valid:
            node = architectures
            prefix: tuple[str, ...] = ()
            for part in parts:
                render_level(node, prefix, part if node.get(part) is None else f"{part} ›")
                child = node[part]
                if child is None:
                    break
                node = child
                prefix = (*prefix, part)
        else:
            render_level(architectures, (), CUSTOM_PATH)
            custom_var.set(canonical)
            custom_entry.grid(row=1, column=0, sticky="ew", pady=(5, 0))
            browse.grid(row=1, column=1, sticky="w", padx=(8, 0), pady=(5, 0))

        def value() -> str:
            if selectors and selectors[0][0].get() == CUSTOM_PATH:
                return custom_var.get()
            values: list[str] = []
            for variable, node, _prefix in selectors:
                choice = variable.get()
                if not choice or choice == CUSTOM_PATH:
                    return ""
                name = choice.removesuffix(" ›")
                values.append(name)
                if node.get(name) is None:
                    return "/".join(values)
            return ""

        return _WidgetField(field, container, value, lambda value: custom_var.set(value))

    def _browse_for_path(self, widget_field: _WidgetField) -> None:
        selected = self._select_file_or_parent()
        if selected is not None:
            widget_field.set_value(selected)

    def _browse_model_path(self, variable: tk.StringVar) -> None:
        selected = self._select_file_or_parent()
        if selected is not None:
            variable.set(selected)

    def _select_file_or_parent(self) -> str | None:
        start = Path.cwd()
        dialog = tk.Toplevel(self.root)
        dialog.title("Select file or folder")
        dialog.configure(background="#639A00")
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.minsize(560, 360)
        browser = tk.Frame(dialog, background="#121a17", highlightthickness=0)
        browser.grid(row=0, column=0, sticky="nsew", padx=1, pady=1)
        current = tk.StringVar(value=str(start))
        result: dict[str, Path | None] = {"path": None}
        items: list[Path] = []
        listbox = tk.Listbox(
            browser,
            activestyle="none",
            background="#0f1714",
            foreground="#f0f7df",
            selectbackground="#639A00",
            selectforeground="#090d0c",
            highlightthickness=1,
            highlightbackground="#639A00",
            highlightcolor="#639A00",
            relief="flat",
            bd=0,
        )

        def refresh(folder: Path) -> None:
            try:
                entries = sorted(folder.iterdir(), key=lambda entry: (not entry.is_dir(), entry.name.lower()))
            except OSError:
                return
            current.set(str(folder))
            items[:] = entries
            listbox.delete(0, "end")
            for entry in entries:
                listbox.insert("end", f"{entry.name}/" if entry.is_dir() else entry.name)

        def highlighted() -> Path | None:
            selected = listbox.curselection()
            return items[selected[0]] if selected else None

        def open_item(_event: object = None) -> None:
            item = highlighted()
            if item is None:
                return
            if item.is_dir():
                refresh(item)
            else:
                result["path"] = item
                dialog.destroy()

        def select() -> None:
            item = highlighted()
            result["path"] = browser_selection(current_folder=Path(current.get()), highlighted=item, highlighted_is_file=bool(item and item.is_file()))
            dialog.destroy()

        def up() -> None:
            folder = Path(current.get())
            if folder.parent != folder:
                refresh(folder.parent)

        dialog.columnconfigure(0, weight=1)
        dialog.rowconfigure(0, weight=1)
        browser.columnconfigure(0, weight=1)
        browser.rowconfigure(1, weight=1)
        top = ttk.Frame(browser, padding=(10, 10, 10, 4))
        top.grid(row=0, column=0, sticky="ew")
        top.columnconfigure(1, weight=1)
        ttk.Button(top, text="Up", command=up).grid(row=0, column=0, padx=(0, 8))
        ttk.Label(top, textvariable=current).grid(row=0, column=1, sticky="w")
        listbox.grid(row=1, column=0, sticky="nsew", padx=10, pady=4)
        listbox.bind("<Double-1>", open_item)
        buttons = ttk.Frame(browser, padding=(10, 4, 10, 10))
        buttons.grid(row=2, column=0, sticky="e")
        ttk.Button(buttons, text="Cancel", command=dialog.destroy).grid(row=0, column=0, padx=(0, 8))
        ttk.Button(buttons, text="Select", command=select).grid(row=0, column=1)
        refresh(start)
        self.root.wait_window(dialog)
        return str(result["path"]) if result["path"] is not None else None

    def _capture_values(self) -> None:
        self.form_values.update(
            {key: widget_field.value() for key, widget_field in self.widget_fields.items()}
        )

    def run(self) -> None:
        if self.running:
            return
        self._capture_values()
        path = self._selected_path()
        try:
            argv = build_argv(self.target.parser(), path, self.form_values)
            validate_argv(self.target.parser(), argv)
        except ArgumentValidationError as exc:
            messagebox.showerror("Invalid command", str(exc), parent=self.root)
            return

        command = [
            sys.executable,
            "-u",
            "-m",
            "maesy_ui.runner",
            self.target.dispatch_module,
            *argv,
        ]
        try:
            self.process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            messagebox.showerror("Unable to run command", str(exc), parent=self.root)
            return

        self.running = True
        self._set_running_state(True)
        self.status_var.set(f"Running {self.target.display_name}…")
        threading.Thread(target=self._stream_process, daemon=True).start()

    def _stream_process(self) -> None:
        assert self.process is not None
        process = self.process
        readers = [
            threading.Thread(target=self._read_stream, args=(stream,), daemon=True)
            for stream in (process.stdout, process.stderr)
            if stream is not None
        ]
        for reader in readers:
            reader.start()
        return_code = process.wait()
        for reader in readers:
            reader.join()
        self.output_queue.put(("done", return_code))

    def _read_stream(self, stream: Any) -> None:
        try:
            for line in iter(stream.readline, ""):
                self.output_queue.put(("output", line))
        finally:
            stream.close()

    def _drain_output(self) -> None:
        try:
            while True:
                kind, payload = self.output_queue.get_nowait()
                if kind == "output" and payload:
                    self._append_output(payload)
                elif kind == "done":
                    self.running = False
                    self.process = None
                    self._set_running_state(False)
                    self.status_var.set(
                        "Finished" if payload == 0 else f"Failed (exit code {payload})"
                    )
        except queue.Empty:
            pass
        self.root.after(50, self._drain_output)

    def _append_output(self, text: str) -> None:
        self.output.configure(state="normal")
        self.output.insert("end", text)
        self.output.see("end")
        self.output.configure(state="disabled")

    def _clear_output(self) -> None:
        self.output.configure(state="normal")
        self.output.delete("1.0", "end")
        self.output.configure(state="disabled")


def main() -> None:
    root = tk.Tk()
    MaesyUiApp(root)
    root.mainloop()
