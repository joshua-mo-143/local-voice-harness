# Service management

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

Stopping or uninstalling Herdr requires explicit confirmation through the option:

```bash
voice-harness services stop --include-herdr
voice-harness services uninstall --include-herdr
```

The small process launchers are Python. The `.service` files remain declarative
systemd units rather than implementing a second process supervisor.
