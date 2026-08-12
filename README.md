# Local Voice Agent Harness

Use your voice to talk to a local assistant, dictate into desktop applications, and
delegate repository work to Herdr-managed Cursor agents without leaving your current
task.

## What it does

- **Voice-driven coding delegation** — hand off GitHub issues, pull requests, Linear
  tickets, and repository tasks to Cursor agents.
- **Local or hosted inference** — run the LLM and TTS locally, use Venice AI for
  either service, or mix the two. Wake-word detection and speech-to-text remain local.
- **Focused context** — refer to a focused GitHub, pull request, or enabled Zendesk
  tab, selected editor text, or an uncommitted diff.
- **Streaming speech and interruption** — hear the first clause while later clauses
  are still being synthesized, and interrupt playback with the wake word.
- **Durable background jobs** — continue working while longer agent tasks run and
  receive their results or questions later.
- **Desktop dictation** — transcribe into the focused application with push-to-talk
  or silence-terminated VAD.
- **Conservative automation** — external content is untrusted, and the harness does
  not automatically commit, push, open pull requests, or remove worktrees.

The harness is designed for a trusted, single-user Linux workstation. It is a
personal project rather than a supported production service; read the
[maintainer note](#note-from-maintainer) and [security notes](docs/security.md) before
granting it access to repositories or external integrations.

## Deployment profiles

LLM and TTS providers are selected independently:

| Profile | LLM | TTS | Speech-to-text | Notes |
| --- | --- | --- | --- | --- |
| All local | Qwen through llama.cpp | Chatterbox Turbo | Parakeet or faster-whisper | Private model inference; requires a capable NVIDIA GPU for the tested configuration |
| Hosted | Venice AI | Venice AI | Local | Lower local model requirements; transcripts sent to the LLM and replies sent to TTS leave the machine |
| Mixed | Local or Venice | Local or Venice | Local | Choose where text generation and speech synthesis run separately |

Cursor, GitHub, Linear, and Venice are external services when enabled. “All local”
describes the voice-model pipeline, not delegated Cursor work or optional
integrations.

## Requirements

The supported platform is Linux x86-64 with systemd user services and PipeWire. The
one-shot Arch/CachyOS installer currently provisions CUDA dictation and therefore
expects an NVIDIA GPU. The tested all-local machine has an RTX 5070 Ti Laptop GPU
with 12 GB VRAM and 32 GB system RAM.

CPU dictation and hosted LLM/TTS configurations are available through the
[manual installation guide](docs/installation.md), but the one-shot installer is
not yet a GPU-free installation path. See the guide for
[compute requirements](docs/installation.md#compute-requirements) and
[external prerequisites](docs/installation.md#external-prerequisites).

The supplied systemd units assume the repository is cloned to
`$HOME/local-voice-harness`.

## Quick start

Clone the repository and run the installer as a file:

```bash
git clone https://github.com/joshua-mo-143/local-voice-harness \
  "$HOME/local-voice-harness"
cd "$HOME/local-voice-harness"
./scripts/install.sh
```

Do not pipe the installer over standard input. It finds the repository from its own
path and may pause for interactive authentication. It asks whether the LLM and TTS
should use `local` or `venice`; the choices can also be supplied non-interactively:

```bash
LLM_PROVIDER=venice TTS_PROVIDER=venice ./scripts/install.sh
```

After installation, select the PipeWire microphone and, when using the local LLM,
the llama.cpp CUDA device:

```bash
wpctl status
voice-harness config set audio.source '<PIPEWIRE_SOURCE_NAME>'
voice-harness config set compute.cuda_device CUDA0
```

Use `voice-harness config set` rather than systemd environment drop-ins for
user-selected providers, devices, audio, integrations, and platform settings. Apply
the reported service restarts, then verify the installation:

```bash
voice-harness doctor
voice-harness services status
voice-harness services audit
```

The installer currently writes the compatible legacy `backends.toml` provider file.
It continues to take precedence over matching values in `config.toml`. If you
selected Venice and want to migrate those values into unified configuration, use:

```bash
voice-harness setup --profile showcase
```

This profile selects Venice for LLM and TTS and preserves the old file as
`backends.toml.migrated`. Store a key with `voice-harness credentials set` when
needed. For interactive configuration that retains an existing legacy provider
choice, run `voice-harness setup` instead. See
[Configuration](docs/configuration.md) for precedence and migration details.

### Install with a coding agent

Clone the repository, open it in a coding agent such as the
[Cursor CLI](https://cursor.com/docs/cli/installation), and provide:

```text
Install this project by running ./scripts/install.sh from the repository root.
It is idempotent and installs system packages, uv environments, models, and
systemd user services. Pause for me to complete any interactive GitHub, Cursor,
or credential prompts. After installation, find my microphone with `wpctl
status` and configure it with `voice-harness config set audio.source ...`.
If I selected the local LLM, confirm its CUDA device with `llama-server
--list-devices` and configure `compute.cuda_device`. Finally run
`voice-harness doctor`, `voice-harness services status`, and
`voice-harness services audit`.
```

For non-Arch systems, CPU dictation, or a partial installation, follow the
[manual installation guide](docs/installation.md).

## Usage

### Wake mode

Say “Hey Jarvis” followed by a request:

```text
Hey Jarvis, what time is it?
Hey Jarvis, ask Cursor to summarize the api-docs repository.
Hey Jarvis, ask Cursor to work on Linear issue API-79.
Hey Jarvis, ask Cursor to work on owner/repository#42.
Hey Jarvis, work on owner/repository#12 and owner/repository#18.
Hey Jarvis, create an issue in this repo about the launcher failing after reboot.
Hey Jarvis, create a Linear ticket in team API about the launcher failing after reboot.
Hey Jarvis, summarize this issue.  # with a GitHub issue focused
Hey Jarvis, summarize this ticket. # with a Zendesk ticket focused (opt-in; see below)
Hey Jarvis, fork this repo and add Venice.  # with a public GitHub repo focused
Hey Jarvis, is Linear enabled?
Hey Jarvis, is the voice harness healthy?
Hey Jarvis, check out this PR and make sure it works.  # with a PR focused
Hey Jarvis, what is the status of that Cursor job?
Hey Jarvis, cancel that Cursor job.
```

The assistant speaks its reply. Say “Hey Jarvis” during playback to interrupt it
and begin another request.

Creating a GitHub issue first produces a title and body draft. The full draft is
displayed, a short summary is spoken, and no issue is created until you answer the
separate confirmation question with a direct yes.

Creating a Linear ticket follows the same preview and direct-confirmation flow. The
team must come from the spoken request or a validated focused Linear team page, and
the authenticated Linear MCP integration performs the write only after confirmation.

Focused browser, editor, and terminal capture is bounded and opt-in. GitHub is
enabled by default; Linear and Zendesk are disabled until configured. See
[Context capture](docs/context-capture.md) and
[Configuration](docs/configuration.md#unified-configuration-configtoml).

### Cursor jobs and questions

Long-running work moves to a durable background job. Spoken requests can check its
status, cancel it, or answer a clarification. The equivalent CLI commands include:

```bash
voice-harness jobs list
voice-harness jobs status
voice-harness jobs reply --job 0123456789ab "Use the existing API."
voice-harness jobs cancel 0123456789ab
```

One explicit request may name multiple GitHub or Linear tickets. Each valid target
becomes an independent job; admission failures for one target do not roll back jobs
that already started. Medium- and high-risk ticket work uses reviewed plans and asks
for approval before implementation. See
[Cursor plan approval](docs/configuration.md#cursor-plan-approval).

### Dictation mode

Bind `voice-harness dictate toggle` to a compositor key for push-to-talk recording.
For example, in i3:

```text
bindsym $mod+d exec --no-startup-id $HOME/.local/bin/voice-harness dictate toggle
```

Press once to start and again to stop. Bind `voice-harness dictate vad` for an
always-on listener that transcribes each utterance after silence; invoke it again to
disable the listener. Sway and Hyprland examples and required compositor environment
imports are documented in [Context capture](docs/context-capture.md).

### Useful commands

```bash
voice-harness config show
voice-harness integrations list
voice-harness integrations doctor
voice-harness vocabulary list
voice-harness replay inspect /path/to/replay.json
voice-harness services logs --follow
```

See the [configuration reference](docs/configuration.md),
[vocabulary guide](docs/vocabulary.md), and
[replay guide](docs/replay.md) for details.

## Service management

Use `voice-harness services <command>` with `install`, `start`, `stop`, `restart`,
`status`, `audit`, `logs`, or `uninstall`. systemd supervises the services and keeps
journal logs. `start` launches dictation and the wake listener; local Qwen and
Chatterbox services remain on demand. Herdr is left running unless
`--include-herdr` is explicitly supplied. See
[Service management](docs/service-management.md).

## Privacy and safety

- Wake detection and transcription run locally. Provider selection determines
  whether transcript and response text is sent to Venice.
- GitHub, Linear, Zendesk, focused application content, and repository diffs are
  captured only through their enabled, bounded context providers.
- Durable job state lives in the private harness state directory; recordings,
  sockets, and worker logs are session-scoped under `$XDG_RUNTIME_DIR`.
- Wake-service journal entries may contain transcripts, responses, repository
  context, and tool results.
- Voice recognition and model routing can be wrong. Inspect agent work before
  committing or publishing it.

Read [Security notes](docs/security.md) and the
[service hardening policy](docs/service-hardening.md) for the complete trust model.

## Documentation

- [Manual installation](docs/installation.md) — profiles, dependencies, models,
  authentication, audio, and services.
- [Configuration](docs/configuration.md) — unified configuration, defaults, legacy
  precedence, integrations, and plan approval.
- [Service management](docs/service-management.md) — service lifecycle and logs.
- [Architecture](docs/architecture.md) — pipeline, routing, jobs, privacy, and
  durability.
- [Context capture](docs/context-capture.md) — browser, editor, terminal, and
  compositor support.
- [Vocabulary and aliases](docs/vocabulary.md) — local transcription and entity
  corrections.
- [Reproducible replay](docs/replay.md) — side-effect-free semantic replay.
- [Troubleshooting](docs/troubleshooting.md) — common failures and diagnostics.
- [Development](docs/development.md) — local launcher and CI-equivalent checks.
- [Security notes](docs/security.md) and
  [service hardening](docs/service-hardening.md) — trust boundaries and residual
  risks.
- [Hardware smoke checks](docs/hardware-smoke.md) — opt-in GPU, audio, and complete
  voice-turn verification.

`docs/durable-storage-migration.md` is an internal design and migration record, not
an installation or operations guide.

## Note from maintainer
Hi! (this note is entirely human-written)

I primarily built this to help me interface with my Cursor agents a bit better. It is annoying to have to go to a Github issue and ask it to work on (or verify) the issue, or do the entire workflow for just gh pr checkout-ing a user's changes and checking locally to make sure everything makes sense. And the same with Linear tickets... and also Zendesk tickets to a degree.

Pretty much all of this code is AI generated. While I am dogfooding this harness (so any bugs are likely to be found and squashed by me) and I'll be making all the major architectural decisions on this personal project, using this software will primarily be at your own risk. I've tried to make the installation as user-friendly as possible, but you may come across unforeseen issues.

To add: I also am not looking at the code - as long as it works, I will likely keep using it (and fixing bugs as I find them). ~~Be prepared to see LLM-made horrors beyond comprehension should you explore the codebase.~~ There will be attempts made to make sure the code is not the programming equivalent of a Lovecraftian horror. However, you may find that the code quality is perhaps not the best quality due to me just wanting to get stuff to work.

## License

MIT
