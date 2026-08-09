# Local Voice Agent Harness

Delegate coding work to Cursor by voice without breaking your flow, with local AI.

## Features

- **Fully local voice pipeline** - wake word ("Hey Jarvis"), speech-to-text,
  conversation, and speech synthesis all run on your own GPU; no cloud services.
- **Delegates coding work to Cursor agents** - hand off GitHub issues and pull
  requests, Linear tickets, and repository tasks to Herdr-managed Cursor agents by
  voice.
- **Screen-aware requests** - reference a focused GitHub/PR/Zendesk tab or your
  editor selection and diff ("summarize this issue", "explain this error").
- **Streaming replies with barge-in** - playback starts on the first clause, and you
  can interrupt mid-sentence just by saying the wake word again.
- **Background jobs** - long tasks keep running and are spoken back to you when they
  finish, so you are not blocked waiting.
- **Push-to-talk dictation** - a keybind transcribes speech straight into the focused
  window, with automatic paste/type per application.
- **Safe by default** - never auto-commits, pushes, or opens PRs; browser and ticket
  content is treated as untrusted, and services run under systemd hardening.

## Installation

Runs on Linux x86-64 with systemd user services, PipeWire, and a CUDA-capable
NVIDIA GPU (tested on an RTX 5070 Ti Laptop, 12 GB VRAM). For full hardware and
software prerequisites, see [Compute requirements](docs/installation.md#compute-requirements)
and [External prerequisites](docs/installation.md#external-prerequisites) in the
installation guide. The supplied systemd units assume the repository is cloned to
`$HOME/local-voice-harness`.

### Quick install

Clone the repository, then run the one-shot installer:

```bash
git clone https://www.github.com/joshua-mo-143/local-voice-harness "$HOME/local-voice-harness" \
  && cd "$HOME/local-voice-harness" \
  && ./scripts/install.sh
```

Do not pipe the script over stdin (`... | bash`); it locates the repository from
its own path and the logins need the terminal. Run it as a file.

The installer asks whether the LLM and TTS should run locally or use Venice AI.
They can be selected independently. For an unattended install, set
`LLM_PROVIDER` and `TTS_PROVIDER` to `local` or `venice`.

### Install with a coding agent

Prefer to let an agent drive it? Clone the repository, open it in your agent
(such as the [Cursor CLI](https://cursor.com/docs/cli/installation)), and paste:

```text
Install this project by running ./scripts/install.sh from the repo root. It is
idempotent and sets up system packages, the uv environments, model downloads,
and the systemd user services. It will pause for interactive GitHub, Cursor, and
Linear logins only if I am not already authenticated; let me complete those in
the terminal. When it finishes, help me set VOICE_HARNESS_SOURCE to my PipeWire
microphone (find it with `wpctl status`) via
`systemctl --user edit voice-harness-wake.service`, and edit
systemd/user/voice-harness-llm.service if my NVIDIA device is not CUDA0. Then run
`voice-harness status` and `voice-harness services audit` to verify.
```

Prefer to run the steps yourself, or not on Arch? The
[manual installation guide](docs/installation.md) breaks down exactly what the
installer does: the uv environments, model downloads, Cursor/Herdr setup, audio
configuration, and service install.

## Usage

### Wake mode

Say “Hey Jarvis” followed by your request:

```text
Hey Jarvis, what time is it?
Hey Jarvis, ask Cursor to summarize the api-docs repository.
Hey Jarvis, ask Cursor to work on Linear issue API-79.
Hey Jarvis, ask Cursor to work on owner/repository#42.
Hey Jarvis, summarize this issue.  # with a GitHub issue focused in Firefox
Hey Jarvis, summarize this ticket. # with a Zendesk ticket focused in Firefox
Hey Jarvis, fork this repo and add Venice.  # with a public GitHub repo focused
Hey Jarvis, ask Cursor to check out this PR and make sure it works.  # with a PR focused
Hey Jarvis, what is the status of that Cursor job?
Hey Jarvis, cancel that Cursor job.
```

The assistant speaks its reply. To interrupt it and ask something else, say “Hey
Jarvis” again while it is talking.

To act on what is on your screen, focus a GitHub, pull request, or Zendesk tab, or
select code in your editor, then refer to it in your request (for example
“summarize this issue” or “explain this error”). See
[Context capture](docs/context-capture.md) for the supported sources and how to
enable them.

### Dictation mode

Bind `voice-harness dictate toggle` to a key in your compositor, for example i3:

```text
bindsym $mod+d exec --no-startup-id /home/joshuam/.local/bin/voice-harness dictate toggle
```

This toggles recording - so in this case to activate recording you press $mod+d; to stop the recording, press $mod+d again.

For always-on, silence-terminated dictation, bind `voice-harness dictate vad` to a
key. The first invocation enables the listener. Each utterance is transcribed after
speech is followed by silence, then the listener rearms for the next utterance.
Invoke the command again to disable the listener. Keybind configuration remains
external to the harness.

The Sway (`bindsym`) and Hyprland (`bind = SUPER, D, exec, ...`) equivalents, plus
the compositor environment setup the wake service needs, are in
[Context capture](docs/context-capture.md).

## Configuration

Runtime behaviour is set through environment variables on the wake service drop-in
(`systemctl --user edit voice-harness-wake.service`) and dictation selectors in
`~/.config/dictation/backend.env`. The [configuration reference](docs/configuration.md)
lists every variable, its default, and where it may be set, plus repository-root and
vocabulary options.

## Service management

Manage the services with `voice-harness services <command>` (`start`, `stop`,
`restart`, `status`, `audit`, `logs`, `install`, `uninstall`); systemd handles
supervision, restarts, and journal logs. `start` brings up dictation and the wake
listener, leaving Qwen and Chatterbox on-demand; Herdr is left running unless you
pass `--include-herdr`. See [Service management](docs/service-management.md) for the
full command reference.

## Documentation

- [Manual installation](docs/installation.md) - step-by-step breakdown of what the installer does.
- [Configuration](docs/configuration.md) - every environment variable, default, and where to set it.
- [Service management](docs/service-management.md) - the `voice-harness services` command reference.
- [Architecture](docs/architecture.md) - pipeline, background jobs, Cursor routing, runtime privacy and durability.
- [Context capture](docs/context-capture.md) - browser and focused editor/terminal context, compositor setup.
- [Vocabulary and entity aliases](docs/vocabulary.md) - the local transcription/routing correction store.
- [Development](docs/development.md) - quality checks, CI matrix, observed performance.
- [Security notes](docs/security.md) - trust model and safe-by-default behavior.
- [Troubleshooting](docs/troubleshooting.md) - common failures and diagnostics.
- [Service hardening](docs/service-hardening.md) - systemd hardening and local trust policy.
- [Hardware smoke checks](docs/hardware-smoke.md) - opt-in GPU/audio/voice-turn checks.

A full list of documentation files can be found in `docs/*`.

## Note from maintainer
Hi! (this note is entirely human-written)

I primarily built this to help me interface with my Cursor agents a bit better. It is annoying to have to go to a Github issue and ask it to work on (or verify) the issue, or do the entire workflow for just gh pr checkout-ing a user's changes and checking locally to make sure everything makes sense. And the same with Linear tickets... and also Zendesk tickets to a degree.

Pretty much all of this code is AI generated. While I am dogfooding this harness (so any bugs are likely to be found and squashed by me) and I'll be making all the major architectural decisions on this personal project, using this software will primarily be at your own risk. I've tried to make the installation as user-friendly as possible, but you may come across unforeseen issues.

To add: I also am not looking at the code - as long as it works, I will likely keep using it (and fixing bugs as I find them). ~~Be prepared to see LLM-made horrors beyond comprehension should you explore the codebase.~~ There will be attempts made to make sure the code is not the programming equivalent of a Lovecraftian horror. However, you may find that the code quality is perhaps not the best quality due to me just wanting to get stuff to work.

## License
MIT
