# Service management

The Python CLI is the user-facing service manager. Core orchestration talks to
platform interfaces for user-service supervision, credential storage, and
notifications. On Linux those interfaces are implemented by systemd user units,
the desktop Secret Service (`secret-tool`), and `notify-send`. Systemd hardening
and unit-audit logic stay inside that systemd implementation.

The Python CLI is the user-facing service manager; systemd remains responsible for
process supervision, dependencies, restarts, and journal logs:

```bash
voice-harness services install
voice-harness services start
voice-harness services stop
voice-harness services restart
voice-harness services status
voice-harness services audit
voice-harness services logs
voice-harness services logs --follow
voice-harness services uninstall
```

`start` launches dictation and the wake listener. Qwen and Chatterbox remain
on-demand. `stop` shuts down the wake listener, model servers, and dictation in a
safe order but leaves Herdr and active Cursor agents running.

Each management command resolves one validated `UserConfig` snapshot. Configured
Herdr paths and timeouts are used by status and shutdown, while audits validate the
effective units against the same snapshot. `restart` reuses one snapshot for its
whole stop/start sequence: always-on services restart immediately, and active
on-demand model services are stopped and pick up configuration on their next use.
Changing from local to hosted providers still reports an active local model service
for this stop-on-next-restart lifecycle.

Stopping or uninstalling Herdr requires explicit confirmation through the option:

```bash
voice-harness services stop --include-herdr
voice-harness services uninstall --include-herdr
```

The small process launchers are Python. The `.service` files remain declarative
systemd units rather than implementing a second process supervisor. They contain
service-owned socket, cache, and runtime paths but no hard-coded provider, model,
device, dictation, audio, integration, or platform choices. Those come from
`config.toml` when each launcher starts. `backend.env` remains a read-only legacy
resolver input; service installation and auditing never rewrite it.

All services use `Restart=on-failure` with a five-starts-per-minute rate limit.
Audits report restart history for a healthy active service without treating that
history alone as a crash loop; a service still activating or failed at the limit is
unhealthy.
