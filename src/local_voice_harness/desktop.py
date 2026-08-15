from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Protocol

from .platform_services import user_services


class DesktopError(RuntimeError):
    pass


DESKTOP_ENVIRONMENT = (
    "DISPLAY",
    "XAUTHORITY",
    "XDG_SESSION_TYPE",
    "WAYLAND_DISPLAY",
    "XDG_CURRENT_DESKTOP",
    "HYPRLAND_INSTANCE_SIGNATURE",
    "SWAYSOCK",
)


@dataclass(frozen=True)
class Window:
    token: str
    window_class: str
    pid: int | None


@dataclass(frozen=True)
class DesktopCapabilities:
    """Detected desktop features; missing APIs are reported, not required."""

    name: str
    session: str
    active_window: bool
    clipboard: bool
    type_text: bool
    send_key: bool
    overlay: bool
    detail: str


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


def _refresh_desktop_environment() -> None:
    """Recover graphical-session variables imported after this service started."""
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        return
    allowed = set(DESKTOP_ENVIRONMENT)
    for name, value in user_services().user_environment().items():
        if name in allowed and value:
            os.environ.setdefault(name, value)


class Desktop(Protocol):
    def active_window(self) -> Window | None: ...

    def has_clipboard(self) -> bool: ...

    def read_clipboard(self) -> tuple[bool, str]: ...

    def write_clipboard(self, text: str) -> bool: ...

    def send_key(self, key: str, *, window: Window | None = None) -> bool: ...

    def type_text(self, text: str, *, window: Window | None = None) -> None: ...

    def capabilities(self) -> DesktopCapabilities: ...


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

    def type_text(self, text: str, *, window: Window | None = None) -> None:
        if shutil.which("xdotool") is None:
            raise DesktopError("xdotool is required to insert recognized text")
        command = ["xdotool", "type", "--clearmodifiers", "--", text]
        if window is not None:
            command = [
                "xdotool",
                "type",
                "--window",
                window.token,
                "--clearmodifiers",
                "--",
                text,
            ]
        process = _run(command)
        if process is None or process.returncode:
            raise DesktopError(
                "could not insert recognized text into the active window"
            )

    def capabilities(self) -> DesktopCapabilities:
        clipboard = self.has_clipboard()
        input_ready = shutil.which("xdotool") is not None
        missing: list[str] = []
        if not input_ready:
            missing.append("xdotool")
        if not clipboard:
            missing.append("xclip")
        if missing:
            detail = (
                "X11 focused-window automation is incomplete; missing "
                + ", ".join(missing)
            )
        else:
            detail = "X11 focused-window automation is available"
        return DesktopCapabilities(
            name="x11",
            session="x11",
            active_window=input_ready,
            clipboard=clipboard,
            type_text=input_ready,
            send_key=input_ready,
            overlay=True,
            detail=detail,
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

    def type_text(self, text: str, *, window: Window | None = None) -> None:
        if shutil.which("wtype") is None:
            raise DesktopError("wtype is required to insert recognized text")
        if window is not None and self.active_window() != window:
            raise DesktopError(
                "could not insert recognized text into the active window"
            )
        process = _run(["wtype", "-"], input_text=text)
        if process is None or process.returncode:
            raise DesktopError(
                "could not insert recognized text into the active window"
            )

    def _wayland_tool_capabilities(
        self,
        *,
        name: str,
        active_window: bool,
        overlay: bool,
        window_tool: str,
    ) -> DesktopCapabilities:
        clipboard = self.has_clipboard()
        input_ready = shutil.which("wtype") is not None
        missing: list[str] = []
        if not active_window:
            missing.append(window_tool)
        if not clipboard:
            missing.append("wl-clipboard")
        if not input_ready:
            missing.append("wtype")
        if missing:
            detail = (
                f"{name} focused-window automation is incomplete; missing "
                + ", ".join(missing)
            )
        else:
            detail = f"{name} focused-window automation is available"
        return DesktopCapabilities(
            name=name,
            session="wayland",
            active_window=active_window,
            clipboard=clipboard,
            type_text=input_ready and active_window,
            send_key=input_ready and active_window,
            overlay=overlay,
            detail=detail,
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

    def capabilities(self) -> DesktopCapabilities:
        return self._wayland_tool_capabilities(
            name="hyprland",
            active_window=shutil.which("hyprctl") is not None,
            overlay=True,
            window_tool="hyprctl",
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

    def capabilities(self) -> DesktopCapabilities:
        return self._wayland_tool_capabilities(
            name="sway",
            active_window=shutil.which("swaymsg") is not None,
            overlay=True,
            window_tool="swaymsg",
        )


def _desktop_tokens(value: str) -> set[str]:
    return {token for token in value.replace(":", ";").split(";") if token}


def _desktop_matches(desktop_name: str, *needles: str) -> bool:
    tokens = _desktop_tokens(desktop_name)
    return any(needle in token for token in tokens for needle in needles)


class DegradedWaylandDesktop(WaylandDesktop):
    """Clipboard-only Wayland desktop when compositor APIs are unavailable."""

    def __init__(self, name: str, detail: str) -> None:
        self._name = name
        self._detail = detail

    def active_window(self) -> Window | None:
        return None

    def send_key(self, key: str, *, window: Window | None = None) -> bool:
        raise DesktopError(self._detail)

    def type_text(self, text: str, *, window: Window | None = None) -> None:
        raise DesktopError(self._detail)

    def capabilities(self) -> DesktopCapabilities:
        clipboard = self.has_clipboard()
        detail = self._detail
        if clipboard:
            detail = f"{detail}; clipboard access is available"
        else:
            detail = f"{detail}; install wl-clipboard for clipboard access"
        return DesktopCapabilities(
            name=self._name,
            session="wayland",
            active_window=False,
            clipboard=clipboard,
            type_text=False,
            send_key=False,
            overlay=False,
            detail=detail,
        )


def get_desktop() -> Desktop | None:
    _refresh_desktop_environment()
    session_type = os.environ.get("XDG_SESSION_TYPE", "").casefold()
    if session_type != "wayland" and not os.environ.get("WAYLAND_DISPLAY"):
        return X11Desktop()

    desktop_name = os.environ.get("XDG_CURRENT_DESKTOP", "").casefold()
    if os.environ.get("HYPRLAND_INSTANCE_SIGNATURE") or "hyprland" in desktop_name:
        return HyprlandDesktop()
    if os.environ.get("SWAYSOCK") or desktop_name == "sway":
        return SwayDesktop()
    if _desktop_matches(desktop_name, "gnome"):
        return DegradedWaylandDesktop(
            "gnome",
            "GNOME Wayland does not expose a supported focused-window or "
            "keyboard-injection API; dictation insertion degrades to stdout",
        )
    if _desktop_matches(desktop_name, "kde", "plasma"):
        return DegradedWaylandDesktop(
            "kde",
            "KDE Plasma Wayland does not expose a supported focused-window or "
            "keyboard-injection API; dictation insertion degrades to stdout",
        )
    return DegradedWaylandDesktop(
        desktop_name or "wayland",
        "this Wayland compositor has no supported focused-window automation; "
        "dictation insertion degrades to stdout",
    )
