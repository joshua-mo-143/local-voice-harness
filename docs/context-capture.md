# Context capture

The harness can enrich a conversational request with bounded, opt-in context from the
focused browser tab or the focused code editor/terminal. All captured content is
treated as untrusted external input, never as instructions, and every capture path
fails closed: on any error or focus change it omits context rather than failing the
underlying voice request.

## Browser context

On X11, Hyprland, and Sway, each new conversational request checks whether Firefox
is focused. The harness briefly selects and copies the address bar, restores the
previous clipboard, and dismisses the address bar without navigating. A focused
GitHub page contributes its URL; a focused issue page also contributes title, state,
body, labels, and recent comments fetched through the authenticated `gh` CLI. A
spoken `owner/repository#number` reference fetches the same bounded issue context
without requiring Firefox to be focused. GitHub access remains required; issue
metadata is persisted only as part of the active job and is not an offline cache. A
focused pull request page adds the same details plus its draft state, source and
target branches, and change summary, and lets a Cursor request check the branch out
locally. Missing tools, unsupported Wayland compositors, focus changes during
capture, and browser or GitHub errors simply omit some or all browser context
without failing the voice request.

### Optional context providers

Optional integrations are supplied by a small registry of context providers,
keyed off the `[integrations]` flags in the unified configuration. Each provider
owns its own URL matching and capture, emits a bounded, provenance-labelled,
untrusted fragment, and is only ever instantiated when its flag is enabled. A
provider that raises is isolated and cannot break an ordinary voice request, and
an unreadable configuration falls back to the disabled defaults.

Zendesk is the first such provider and is **disabled by default on fresh
installations**. Enable it with `[integrations] zendesk = true` in `config.toml`
or `VOICE_HARNESS_INTEGRATION_ZENDESK=1`; existing installations preserve prior
behaviour by setting the same flag. While it is disabled, Zendesk URLs are never
inspected and no page text is copied. When enabled, a focused
`https://<tenant>.zendesk.com/agent/tickets/<number>` page contributes its URL,
tenant, ticket number, and bounded rendered page text copied from the
authenticated browser session; no Zendesk API credentials are required. Only text
currently loaded and selectable in the page is available, so collapsed or unloaded
comments may be absent, and the page content is treated as untrusted input.

## Focused editor and terminal context

When the focused window is a supported code editor (Cursor, VS Code, VSCodium) or a
terminal, a request that explicitly refers to focused content — such as "explain this
error" or "fix this code" — additionally captures bounded, opt-in context from that
application. Two sources are supported:

- **Selected text** (`selection`): the current editor or terminal selection, copied
  through the clipboard while preserving both the previous clipboard contents and
  window focus. Bounded to 4,000 characters.
- **Uncommitted git diff** (`git_diff`): `git diff` for the repository containing the
  focused window's working directory. Bounded to 8,000 characters.

The combined focused-app context is capped at 12,000 characters
(`VOICE_HARNESS_FOCUSED_APP_MAX_CHARS`) and is labelled as untrusted external input,
never as instructions. Capture is fenced by a deny-list of sensitive or unsupported
window classes (password managers, secret stores, and RuneLite by default; override
with `VOICE_HARNESS_FOCUSED_APP_DENY`) and can be disabled entirely with
`VOICE_HARNESS_FOCUSED_APP_CONTEXT=0`. It fails closed — omitting context — when focus
changes mid-capture, the compositor or application is unsupported, a source exceeds its
size limit, or any capture step errors, and never fails the underlying voice request.
There is no screenshot, OCR, or continuous screen monitoring; only these explicit,
bounded pulls.

## Compositor environment for the wake service

Native Wayland automation is supported on Hyprland and Sway. It uses `hyprctl` or
`swaymsg` to identify the focused window, `wl-copy`/`wl-paste` for the clipboard,
and `wtype` for keyboard input. GNOME, KDE Plasma, and other compositors are not
currently supported; set `DICTATION_INJECT=stdout` if focused-window insertion is
not required there.

Automatic dictation injection routes by focused application: native Herdr delivery
for Cursor panes, simulated typing for terminals, and clipboard paste for other
graphical applications. Dictation is blocked while RuneLite is focused because
generated input may violate Jagex's rules.

The wake service needs the compositor environment to collect Firefox context.
Import it into the systemd user manager from compositor startup. For Sway:

```text
exec_always systemctl --user import-environment WAYLAND_DISPLAY XDG_CURRENT_DESKTOP SWAYSOCK
```

For Hyprland:

```text
exec-once = systemctl --user import-environment WAYLAND_DISPLAY XDG_CURRENT_DESKTOP HYPRLAND_INSTANCE_SIGNATURE
```

Restart `voice-harness-wake.service` after adding the import. Dictation commands
launched directly by compositor keybindings already inherit the required
environment.
