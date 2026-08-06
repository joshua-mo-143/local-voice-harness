from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Protocol


class DesktopError(RuntimeError):
    pass


@dataclass(frozen=True)
class Window:
    token: str
    window_class: str
    pid: int | None


def _run(
    command: list[str],
    *,
    input_text: str | None = None,
) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(
            command,
            input=input_text,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None


def _write_clipboard(command: list[str], text: str) -> bool:
    try:
        process = subprocess.run(
            command,
            input=text,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return process.returncode == 0


def _integer(value: object) -> int | None:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


class Desktop(Protocol):
    def active_window(self) -> Window | None: ...

    def has_clipboard(self) -> bool: ...

    def read_clipboard(self) -> tuple[bool, str]: ...

    def write_clipboard(self, text: str) -> bool: ...

    def send_key(self, key: str, *, window: Window | None = None) -> bool: ...

    def type_text(self, text: str) -> None: ...


class X11Desktop:
    def active_window(self) -> Window | None:
        if shutil.which("xdotool") is None:
            return None
        active = _run(["xdotool", "getactivewindow"])
        if active is None or active.returncode:
            return None
        token = active.stdout.strip()
        if not token.isdigit():
            return None
        window_class = _run(["xdotool", "getwindowclassname", token])
        pid = _run(["xdotool", "getwindowpid", token])
        return Window(
            token=token,
            window_class=(
                window_class.stdout.strip().casefold()
                if window_class is not None and window_class.returncode == 0
                else ""
            ),
            pid=(
                _integer(pid.stdout.strip())
                if pid is not None and pid.returncode == 0
                else None
            ),
        )

    def read_clipboard(self) -> tuple[bool, str]:
        if shutil.which("xclip") is None:
            return False, ""
        process = _run(["xclip", "-selection", "clipboard", "-out"])
        if process is None or process.returncode:
            return False, ""
        return True, process.stdout

    def has_clipboard(self) -> bool:
        return shutil.which("xclip") is not None

    def write_clipboard(self, text: str) -> bool:
        if shutil.which("xclip") is None:
            raise DesktopError("xclip is required to paste recognized text")
        return _write_clipboard(["xclip", "-selection", "clipboard"], text)

    def send_key(self, key: str, *, window: Window | None = None) -> bool:
        if shutil.which("xdotool") is None:
            raise DesktopError("xdotool is required to insert recognized text")
        command = ["xdotool", "key"]
        if window is not None:
            command.extend(("--window", window.token))
        command.extend(("--clearmodifiers", key))
        process = _run(command)
        return process is not None and process.returncode == 0

    def type_text(self, text: str) -> None:
        if shutil.which("xdotool") is None:
            raise DesktopError("xdotool is required to insert recognized text")
        process = _run(["xdotool", "type", "--clearmodifiers", "--", text])
        if process is None or process.returncode:
            raise DesktopError(
                "could not insert recognized text into the active window"
            )


class WaylandDesktop:
    def active_window(self) -> Window | None:
        raise NotImplementedError

    def read_clipboard(self) -> tuple[bool, str]:
        if shutil.which("wl-paste") is None:
            return False, ""
        process = _run(["wl-paste", "--no-newline"])
        if process is None or process.returncode:
            return False, ""
        return True, process.stdout

    def has_clipboard(self) -> bool:
        return (
            shutil.which("wl-copy") is not None and shutil.which("wl-paste") is not None
        )

    def write_clipboard(self, text: str) -> bool:
        if shutil.which("wl-copy") is None:
            raise DesktopError("wl-copy is required to paste recognized text")
        return _write_clipboard(["wl-copy"], text)

    def send_key(self, key: str, *, window: Window | None = None) -> bool:
        if shutil.which("wtype") is None:
            raise DesktopError("wtype is required to insert recognized text")
        if window is not None and self.active_window() != window:
            return False
        parts = key.split("+")
        modifiers = [part.casefold() for part in parts[:-1]]
        named_key = parts[-1]
        command = ["wtype"]
        for modifier in modifiers:
            command.extend(("-M", modifier))
        command.extend(("-k", named_key))
        for modifier in reversed(modifiers):
            command.extend(("-m", modifier))
        process = _run(command)
        return process is not None and process.returncode == 0

    def type_text(self, text: str) -> None:
        if shutil.which("wtype") is None:
            raise DesktopError("wtype is required to insert recognized text")
        process = _run(["wtype", "-"], input_text=text)
        if process is None or process.returncode:
            raise DesktopError(
                "could not insert recognized text into the active window"
            )


class HyprlandDesktop(WaylandDesktop):
    def active_window(self) -> Window | None:
        if shutil.which("hyprctl") is None:
            return None
        process = _run(["hyprctl", "-j", "activewindow"])
        if process is None or process.returncode:
            return None
        try:
            value = json.loads(process.stdout)
        except json.JSONDecodeError:
            return None
        if not isinstance(value, dict):
            return None
        token = str(value.get("address") or "")
        if not token:
            return None
        return Window(
            token=token,
            window_class=str(
                value.get("class") or value.get("initialClass") or ""
            ).casefold(),
            pid=_integer(value.get("pid")),
        )


def _focused_sway_node(value: object) -> dict[str, object] | None:
    if not isinstance(value, dict):
        return None
    if value.get("focused") is True:
        return value
    for collection in ("nodes", "floating_nodes"):
        children = value.get(collection)
        if not isinstance(children, list):
            continue
        for child in children:
            focused = _focused_sway_node(child)
            if focused is not None:
                return focused
    return None


class SwayDesktop(WaylandDesktop):
    def active_window(self) -> Window | None:
        if shutil.which("swaymsg") is None:
            return None
        process = _run(["swaymsg", "-t", "get_tree"])
        if process is None or process.returncode:
            return None
        try:
            node = _focused_sway_node(json.loads(process.stdout))
        except json.JSONDecodeError:
            return None
        if node is None:
            return None
        token = str(node.get("id") or "")
        if not token:
            return None
        properties = node.get("window_properties")
        xwayland_class = (
            str(properties.get("class") or "") if isinstance(properties, dict) else ""
        )
        return Window(
            token=token,
            window_class=str(node.get("app_id") or xwayland_class).casefold(),
            pid=_integer(node.get("pid")),
        )


def get_desktop() -> Desktop | None:
    session_type = os.environ.get("XDG_SESSION_TYPE", "").casefold()
    if session_type != "wayland" and not os.environ.get("WAYLAND_DISPLAY"):
        return X11Desktop()

    desktop_name = os.environ.get("XDG_CURRENT_DESKTOP", "").casefold()
    if os.environ.get("HYPRLAND_INSTANCE_SIGNATURE") or "hyprland" in desktop_name:
        return HyprlandDesktop()
    if os.environ.get("SWAYSOCK") or desktop_name == "sway":
        return SwayDesktop()
    return None
