## Troubleshooting

Wake listener will not start:

```bash
voice-harness services status
voice-harness services logs
```

CUDA library or model errors:

```bash
journalctl --user -u dictation.service -u voice-harness-tts.service -n 100
nvidia-smi
```

If dictation reports a missing Python module, ensure the installed extra matches
the backend and device: use `dictation` for CPU Parakeet, `dictation-cuda` for
CUDA Parakeet, or `dictation-whisper` for faster-whisper, then restart
`dictation.service`. Unknown backend or device names are rejected at startup.
Explicit `cuda` also fails at startup when the selected runtime cannot see a CUDA
device. Use `voice-harness config set compute.dictation_device cpu` for a strict
CPU path that does not probe CUDA or search NVIDIA library directories.

Herdr/Cursor failures:

```bash
herdr status server
herdr agent list
agent status
agent mcp list
```

Job deletion reports unresolved quarantine evidence:

```fish
voice-harness jobs quarantine list
voice-harness jobs quarantine list --all --json
```

The listing shows the quarantined payload and metadata paths plus any recorded
worker identity, Herdr target, or worktree reservation. Verify those external
resources no longer exist before releasing their fences. Then record what was
checked and retry deletion:

```fish
voice-harness jobs quarantine acknowledge aaaaaaaaaaaa \
  --reason "worker exited and no Herdr target or worktree remains"
voice-harness jobs nuke
```

Acknowledgement requires typing `acknowledge` and writes a hash-bound resolution
tombstone. It does not delete the quarantined payload or metadata. Do not
acknowledge evidence merely to silence the warning: unresolved worker, agent,
fork, or worktree operations must remain fenced until manually reconciled.

SQLite storage or migration failures are reported by:

```fish
voice-harness doctor
```

The jobs diagnostics include the exact `jobs.sqlite3` path, database schema,
migration status, integrity result, and unresolved import-failure count. Diagnostics
are read-only and do not retry migration or quarantine files. If the database cannot
be opened or fails its integrity check, stop the services and preserve the complete
jobs directory before attempting recovery:

```fish
voice-harness services stop
cp -a ~/.local/state/voice-harness/jobs \
  ~/.local/state/voice-harness/jobs.forensics-(date +%Y%m%d-%H%M%S)
```

Restore only a complete backup containing the database, WAL companions (if present),
`.artifacts`, `.quarantine`, and archived JSON inputs. Never delete WAL files,
reservations, or delivery claims individually; doing so can release a live target or
lose an undelivered result. See `docs/durable-storage-migration.md` for the full
backup and rollback limits.

Wrong llama.cpp GPU:

```bash
llama-server --list-devices
systemctl --user edit voice-harness-llm.service
```

After changing shipped units and intentionally adopting the bundled dictation unit:

```fish
voice-harness services install --force --replace-dictation
voice-harness services audit
voice-harness services restart
```
