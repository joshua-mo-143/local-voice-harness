class HarnessError(RuntimeError):
    """User-facing harness failure."""


class NoSpeechError(HarnessError):
    """The speech-to-text backend found no recognizable speech."""
