class HarnessError(RuntimeError):
    """User-facing harness failure."""


class NoSpeechError(HarnessError):
    """The speech-to-text backend found no recognizable speech."""


class SpeechDeliveryError(HarnessError):
    """Spoken delivery failed after a successful textual response."""
