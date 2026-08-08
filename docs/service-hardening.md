# Service hardening and local trust policy

The shipped units target the documented single-user, 32 GB CUDA workstation. The
controls below reduce accidental exposure and post-compromise reach while retaining
the capabilities each service actually uses. They do not make model processes or
external tools safe to run on a hostile multi-user machine.

## Capability model

| Service | Filesystem and home | Network | Devices/session | Processes and runtime |
| --- | --- | --- | --- | --- |
| `dictation` | Reads its project, virtual environment, optional backend config, and request WAVs. Home is read-only except the managed Hugging Face cache. `%t` remains writable to preserve the standalone-compatible `%t/dictation.sock`. | Unix sockets, IPv4/IPv6 for Hugging Face downloads, and netlink for CUDA discovery. | CUDA devices stay visible. The application does not intentionally use PipeWire or desktop APIs, but AF_UNIX and `%t` mean those same-user session endpoints are not an isolation boundary. | The launcher replaces itself with the server. Temporary and CUDA cache files use `%t/dictation`; the compatibility socket remains mode 0600 at `%t/dictation.sock`. |
| `voice-harness-llm` | Reads `llama-server` and the GGUF under the project. Home is read-only. Its writable CUDA cache is `%t/voice-harness-llm/cuda-cache`. | Unix, IPv4, and netlink only. `llama-server` is required to bind `127.0.0.1:8090`. AF_UNIX still permits connections to accessible same-user session sockets. | NVIDIA devices remain visible; the service does not intentionally use audio or desktop APIs. | No application subprocesses. Its only managed writable runtime directory is mode 0700. |
| `voice-harness-tts` | Reads both project environments, backend configuration, the offline Hugging Face cache, and an optional reference WAV. Home is read-only; generated audio uses `%t/voice-harness`, the fixed compatibility socket is `%t/voice-harness-tts.sock`, and the CUDA cache is `%t/voice-harness-tts/cuda-cache`. | Unix sockets include the desktop Secret Service for Venice credentials; IPv4/IPv6 permit the optional Venice backend, and netlink permits CUDA discovery. Local Chatterbox keeps Hugging Face explicitly offline. | NVIDIA devices remain visible. PipeWire playback is performed by clients, not this server. | Uses request threads and invokes an installed Secret Service client for Venice credential lookup. Separate mode-0700 `voice-harness` and `voice-harness-tts` runtime directories are preserved across on-demand stops. |
| `voice-harness-wake` | Reads and writes repositories, worktrees, durable job state in the mode-0700 `StateDirectory=voice-harness`, GitHub/Cursor/Herdr configuration, and user-selected clone destinations. Broad home access is therefore intentional. System paths remain read-only. | Unix sockets, IPv4/IPv6, and netlink support local model services, GitHub/Git/Herdr, DNS, and network inspection by child tools. | PipeWire and desktop/compositor sockets remain usable. Direct devices are hidden because `pw-record`/`pw-play` use PipeWire rather than ALSA. A private `/tmp` is not used because X11 may rely on `/tmp/.X11-unix`. | Starts audio tools, desktop helpers, `gh`, Git, Herdr/systemd commands, and detached job workers. It receives the largest task allowance and shares the mode-0700 voice runtime directory; worker logs remain transient there. |

All services use a `0077` umask, an empty capability set, no-new-privileges,
read-only system paths, kernel/control-group protections, native system-call
architecture, core-dump suppression, bounded file descriptors, and a five-starts-per-
minute rate limit. Namespace creation is disabled. GPU services deliberately do not
use `PrivateDevices`; CUDA requires NVIDIA device nodes. JIT-sensitive
`MemoryDenyWriteExecute`, broad syscall filters, device allowlists, and IP firewall
directives are omitted because their compatibility with CUDA, desktop tools, and
unprivileged user managers is not established.

AF_UNIX is not treated as model-service isolation. The user runtime directory contains
PipeWire, D-Bus, compositor, notification, Herdr, and other same-UID sockets. Path and
mode protections narrow accidental file access, but any accessible same-user session
endpoint remains trusted. Brittle masks are not used because they would block required
CUDA discovery or the standalone dictation protocol.

No shipped unit permits `EnvironmentFile`; the installed audit checks both the
effective unit/drop-in text and systemd's `EnvironmentFiles` property. Required
socket, cache, runtime, and service-control variables must retain their shipped
values. Only `voice-harness-wake.service` accepts optional drop-in variables: the
README-listed audio, voice, wake/barge-in, playback, worker timing, Herdr/repository,
dictation-injection, and replacement settings. Numeric ranges, enums, durations,
absolute paths, and GitHub-root containment are validated. Unknown keys and process
injection or service-boundary overrides such as `LD_PRELOAD`, socket/cache/runtime
paths, and LLM host/bind controls fail the audit.

Venice credentials are stored in the desktop Secret Service through `oo7-cli` or
libsecret's `secret-tool`, not in an application file, environment variable, command
argument, or unit. Storage sends the key over standard input; status and diagnostic output never
print it. Secret Service protects the key at rest and can lock it with the desktop
session, but retrieval necessarily places it briefly in process memory and does not
isolate it from a process already controlling the same unlocked user session.

The memory high/max pairs are 4/6 GB for dictation, 6/8 GB for llama.cpp, 8/10 GB for
TTS, and 3/4 GB for wake and its workers. The 28 GB aggregate maxima leave room on
the tested 32 GB host for the desktop, PipeWire, Herdr, and the kernel. `TasksMax` is
256 for llama.cpp, 512 for each Python model service, and 1024 for the subprocess-
heavy wake service. These are safety ceilings, not measured minimum requirements.
Swap remains unlimited to avoid introducing an unmeasured CUDA cold-start failure;
`OOMPolicy=stop` and `OOMScoreAdjust=200` make the effective OOM behavior explicit.

## Unauthenticated llama.cpp policy

The endpoint is intentionally unauthenticated and bound to IPv4 loopback. Clients in
the same Unix account are trusted to submit prompts and consume GPU time. A static
token embedded in every local client and unit would not separate same-user processes
and would add fragile secret distribution, so none is added.

Loopback is host-local, not UID-scoped: another login account on the same machine may
also be able to connect. The supported deployment is a trusted single-user
workstation. Do not use this configuration on a mutually untrusted multi-user host;
use a Unix-socket credential-checking proxy or another per-user transport there.

## Residual risks and host verification

- CUDA, ONNX Runtime, Chatterbox, llama.cpp, NVIDIA driver behavior, and live Venice
  API behavior cannot be exercised in CI. CUDA cold starts and remote LLM/TTS latency
  remain pending human host verification.
- The dictation cache is writable because a clean installation downloads its model.
  Pre-provisioning a read-only cache would permit a narrower policy.
- The wake service retains broad same-user authority by design. Its filesystem,
  desktop, GitHub, Herdr, and worker permissions are not a boundary against untrusted
  request content. Trusted-request/external-content authorization is handled in a
  separate stacked change.
- The full same-user runtime/session remains trusted. AF_UNIX cannot distinguish a
  legitimate local client from another process running as the same UID.
- `ProtectProc=invisible`, private device visibility for wake, address-family
  restrictions, and resource ceilings require real desktop/audio/GPU validation.

## Pending human host-smoke record

None of the following checks has been run by CI or during this implementation pass.
After intentionally installing the candidate units on the target host, record each
row as `PASS` or `FAIL` with a journal excerpt or observation:

| Check | Result | Evidence |
| --- | --- | --- |
| Effective installed-unit audit | PENDING | |
| Denied system/home write probes | PENDING | |
| Denied address-family probes | PENDING | |
| Dictation CUDA cold start/cache | PENDING | |
| llama.cpp CUDA cold start/cache and loopback bind | PENDING | |
| TTS CUDA cold start/cache and optional reference WAV | PENDING | |
| PipeWire capture/playback and desktop helpers | PENDING | |
| GitHub/Herdr worker dispatch and completion | PENDING | |
| Memory/task/cgroup/OOM observation | PENDING | |
| Cleanup and restart-rate state | PENDING | |

Baseline and read-only effective audit:

```fish
uv run python -m local_voice_harness.service_units --require-systemd-analyze
# Require rooted staged user context when the installed systemd supports it:
uv run python -m local_voice_harness.service_units \
  --require-systemd-analyze --require-user-context
# This is a separate direct user-context check, not a substitute for staged output:
systemd-analyze --user verify systemd/user/*.service
systemd-analyze verify systemd/user/*.service
# This intentionally replaces a preserved standalone dictation unit. Review and
# migrate any backend.env/custom unit settings before running it.
voice-harness services install --force --replace-dictation
systemctl --user daemon-reload
voice-harness services audit
```

Without `--replace-dictation`, the installer deliberately preserves a differing
standalone `dictation.service`; that preserved unit has not adopted this hardening
policy and may still load an unrestricted `EnvironmentFile`. Treat a successful
post-install audit as mandatory evidence that the shipped hardened unit and effective
drop-ins are active.

The staged verifier prints a structured context result. On systemd versions that
reject `--user` together with `--root`, it labels rooted user context `unsupported`
and may perform a distinct staged system-context syntax check. That fallback is never
reported as user-context verification; `--require-user-context` fails in this case.

Denied-operation probes use transient throwaway units with the same relevant
directives; success means the Python operation fails with `PermissionError` or
`EPERM`. They do not alter the installed units:

```fish
systemd-run --user --wait --collect --pipe --unit=voice-harness-write-probe \
  --property=ProtectSystem=strict --property=ProtectHome=read-only \
  /usr/bin/python -c 'open("/etc/voice-harness-probe", "w")'

```

For CUDA cold starts, stop the model services, remove only their disposable runtime
caches, then start them individually. Do not remove model caches under home:

```fish
systemctl --user stop dictation.service voice-harness-llm.service voice-harness-tts.service
rm -rf \
  "$XDG_RUNTIME_DIR/dictation/cuda-cache" \
  "$XDG_RUNTIME_DIR/voice-harness-llm/cuda-cache" \
  "$XDG_RUNTIME_DIR/voice-harness-tts/cuda-cache"
systemctl --user start dictation.service voice-harness-llm.service voice-harness-tts.service
systemctl --user is-active dictation.service voice-harness-llm.service voice-harness-tts.service
curl --fail http://127.0.0.1:8090/health
ss --tcp --listening --numeric | string match '*127.0.0.1:8090*'
nvidia-smi
test -d "$XDG_RUNTIME_DIR/dictation/cuda-cache"
test -d "$XDG_RUNTIME_DIR/voice-harness-llm/cuda-cache"
test -d "$XDG_RUNTIME_DIR/voice-harness-tts/cuda-cache"
```

Exercise PipeWire and desktop/session helpers before dispatching a read-only worker:

```fish
pw-record --version
pw-play --version
wpctl status
notify-send "Voice harness hardening smoke"
voice-harness begin
# Speak a short sentence, then:
voice-harness end
voice-harness services start
voice-harness text "Reply with one short sentence."
voice-harness text "Use Cursor to inspect this repository and report its branch without changing files."
voice-harness status
herdr agent list
```

Observe cgroup pressure and OOM/restart state without intentionally exhausting memory:

```fish
journalctl --user -u dictation.service -u voice-harness-llm.service -u voice-harness-tts.service -u voice-harness-wake.service -n 200
systemctl --user show dictation.service voice-harness-llm.service voice-harness-tts.service voice-harness-wake.service \
  --property=MemoryCurrent --property=MemoryPeak --property=MemoryHigh \
  --property=MemoryMax --property=TasksCurrent --property=TasksMax \
  --property=NRestarts --property=Result
oomctl
voice-harness services stop
```

Also complete the microphone, playback, desktop-context, wake/barge-in, GitHub,
Herdr, and cleanup checks in [the hardware smoke guide](hardware-smoke.md). Verify
that no service is restart-looping, no model falls back unexpectedly to CPU, TTS can
read an optional reference WAV, first-start dictation can populate its cache, browser
automation still works on X11 or the configured Wayland compositor, and detached
workers can finish after the foreground timeout.
