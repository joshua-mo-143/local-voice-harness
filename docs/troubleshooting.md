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
`DICTATION_BACKEND`: use `dictation` for Parakeet or `dictation-whisper` for
faster-whisper, then restart `dictation.service`. Unknown backend names are rejected
at startup.

Herdr/Cursor failures:

```bash
herdr status server
herdr agent list
agent status
agent mcp list
```

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
