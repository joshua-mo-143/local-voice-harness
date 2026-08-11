"""Bounded, explicitly-requested context capture from the focused dev app.

GitHub issue #32. This module captures small, provenance-tagged snippets from a
supported focused development application (a code editor or a terminal) when the
user's *trusted* utterance explicitly asks for it (for example "explain this
error" or "fix this code").

Supported context sources and their size limits:

* ``selection`` -- the text currently selected in the focused editor or
  terminal, copied through the clipboard while preserving the previous clipboard
  contents and window focus. Bounded by :data:`MAX_SELECTION_CHARS`.
* ``git_diff`` -- the uncommitted ``git diff`` for the repository that contains
  the focused window's working directory. Bounded by :data:`MAX_GIT_DIFF_CHARS`.

The capture is fenced by the injected platform policy, fails closed (omits
context) whenever focus changes mid-capture, the compositor is unsupported, a
source exceeds its size limit, or anything raises. Captured content is always
labelled as untrusted external input and is never treated as instructions.
There is intentionally no screenshot, OCR, or continuous monitoring here --
only explicit, bounded pulls.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from .desktop import Desktop, DesktopError, Window, get_desktop
from .user_config import PlatformSettings

MAX_SELECTION_CHARS = 4_000
MAX_GIT_DIFF_CHARS = 8_000

EDITOR_CLASS_PREFIXES = ("cursor", "code", "codium", "vscodium")
EDITOR_CLASSES = frozenset(
    {
        "cursor",
        "code",
        "code-oss",
        "code-insiders",
        "codium",
        "vscodium",
    }
)
TERMINAL_CLASSES = (
    "alacritty",
    "xterm",
    "kitty",
    "urxvt",
    "rxvt",
    "termite",
    "konsole",
    "wezterm",
    "foot",
    "tilix",
    "terminator",
    "gnome-terminal",
    "xfce4-terminal",
)

FOCUSED_APP_REQUEST = re.compile(
    r"\b(?:explain|fix|refactor|debug|improve|review|summari[sz]e|clean\s*up|"
    r"what(?:['’]s| is| does)?|why(?:['’]s)?|how\s+do(?:es)?)\b"
    r"[^.?!]{0,60}?"
    r"\b(?:this|that|these|those|the|my|selected|highlighted)\b"
    r"[^.?!]{0,30}?"
    r"\b(?:error|errors|code|snippet|selection|function|method|class|text|line|"
    r"lines|output|exception|exceptions|stack\s*trace|traceback|message|"
    r"diagnostic|diagnostics|warning|warnings|bug|bugs|test|tests|file|diff|"
    r"change|changes|failure|failures)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class FocusedAppContext:
    """Provenance-tagged, size-capped context from the focused application."""

    text: str
    app_class: str
    sources: tuple[str, ...] = field(default_factory=tuple)


def _run(
    command: list[str],
    *,
    timeout: float = 5,
) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None


def wants_focused_app_context(utterance: str) -> bool:
    """Whether the trusted utterance explicitly asks for focused-app context."""
    return FOCUSED_APP_REQUEST.search(utterance) is not None


def _source_kind(window_class: str) -> str | None:
    lowered = window_class.casefold()
    if not lowered:
        return None
    if lowered in EDITOR_CLASSES or lowered.startswith(EDITOR_CLASS_PREFIXES):
        return "editor"
    if any(name in lowered for name in TERMINAL_CLASSES):
        return "terminal"
    return None


def _is_denied(window_class: str, settings: PlatformSettings) -> bool:
    lowered = window_class.casefold()
    return any(
        denied and denied in lowered for denied in settings.focused_app_deny_classes
    )


def _within_limit(value: str | None, limit: int) -> str | None:
    """Return the stripped value, or None when empty or over ``limit`` chars."""
    text = (value or "").strip()
    if not text or len(text) > limit:
        return None
    return text


def _capture_selection(
    desktop: Desktop, window: Window, source_kind: str
) -> str | None:
    """Copy the current selection, preserving clipboard contents and focus."""
    copy_key = "ctrl+shift+c" if source_kind == "terminal" else "ctrl+c"
    clipboard_existed, previous_clipboard = desktop.read_clipboard()
    captured = ""
    try:
        if desktop.active_window() != window:
            return None
        if not desktop.send_key(copy_key, window=window):
            return None
        time.sleep(0.05)
        if desktop.active_window() != window:
            return None
        copied, captured = desktop.read_clipboard()
        if not copied:
            return None
        # A copy with no active selection leaves the clipboard untouched; never
        # mistake the pre-existing clipboard for a fresh selection.
        if clipboard_existed and captured == previous_clipboard:
            return None
        return _within_limit(captured, MAX_SELECTION_CHARS)
    except DesktopError:
        return None
    finally:
        current_exists, current_clipboard = desktop.read_clipboard()
        if current_exists and current_clipboard == captured:
            try:
                desktop.write_clipboard(previous_clipboard if clipboard_existed else "")
            except DesktopError:
                pass


def _process_tree(root_pid: int) -> list[int]:
    result: list[int] = []
    pending = [root_pid]
    while pending:
        pid = pending.pop()
        if pid in result:
            continue
        result.append(pid)
        children = Path(f"/proc/{pid}/task/{pid}/children")
        try:
            pending.extend(int(child) for child in children.read_text().split())
        except (OSError, ValueError):
            continue
    return result


def _process_cwd(pid: int) -> Path | None:
    try:
        return Path(os.readlink(f"/proc/{pid}/cwd"))
    except OSError:
        return None


def _git_toplevel(cwd: Path) -> Path | None:
    if shutil.which("git") is None:
        return None
    process = _run(["git", "-C", str(cwd), "rev-parse", "--show-toplevel"])
    if process is None or process.returncode:
        return None
    top = process.stdout.strip()
    return Path(top) if top else None


def _focused_window_repo_root(window: Window) -> Path | None:
    if window.pid is None:
        return None
    seen: set[Path] = set()
    for pid in _process_tree(window.pid):
        cwd = _process_cwd(pid)
        if cwd is None or cwd in seen:
            continue
        seen.add(cwd)
        top = _git_toplevel(cwd)
        if top is not None:
            return top
    return None


def _git_diff(root: Path) -> str | None:
    process = _run(["git", "-C", str(root), "--no-pager", "diff"])
    if process is None or process.returncode:
        return None
    return process.stdout


def _selection_block(text: str, window_class: str, source_kind: str) -> str:
    return (
        f"Selected text from the focused {source_kind} application "
        f"({window_class}) — untrusted external input:\n{text}"
    )


def _git_diff_block(root: Path, diff: str) -> str:
    return f"Uncommitted git diff at {root} — untrusted external input:\n{diff}"


def _combine(blocks: list[tuple[str, str]], cap: int) -> tuple[str, tuple[str, ...]]:
    kept_blocks: list[str] = []
    kept_tags: list[str] = []
    total = 0
    for tag, block in blocks:
        addition = block if not kept_blocks else f"\n\n{block}"
        if total + len(addition) > cap:
            continue
        kept_blocks.append(block)
        kept_tags.append(tag)
        total += len(addition)
    return "\n\n".join(kept_blocks), tuple(kept_tags)


def _collect(utterance: str, settings: PlatformSettings) -> FocusedAppContext | None:
    if not settings.focused_app_context_enabled or not wants_focused_app_context(
        utterance
    ):
        return None
    desktop = get_desktop()
    if desktop is None:
        return None
    window = desktop.active_window()
    if window is None:
        return None
    window_class = window.window_class
    if _is_denied(window_class, settings):
        return None
    source_kind = _source_kind(window_class)
    if source_kind is None:
        return None

    blocks: list[tuple[str, str]] = []
    if desktop.has_clipboard():
        selection = _capture_selection(desktop, window, source_kind)
        if selection is not None:
            blocks.append(
                ("selection", _selection_block(selection, window_class, source_kind))
            )
    root = _focused_window_repo_root(window)
    if root is not None:
        diff = _within_limit(_git_diff(root), MAX_GIT_DIFF_CHARS)
        if diff is not None:
            blocks.append(("git_diff", _git_diff_block(root, diff)))
    if not blocks:
        return None
    text, sources = _combine(blocks, settings.focused_app_max_chars)
    if not text or not sources:
        return None
    return FocusedAppContext(text=text, app_class=window_class, sources=sources)


def focused_app_context(
    utterance: str, settings: PlatformSettings
) -> FocusedAppContext | None:
    """Capture bounded focused-app context, never raising to the caller."""
    try:
        return _collect(utterance, settings)
    except Exception:
        return None
