from __future__ import annotations

from pathlib import Path

from ..errors import HarnessError

WAKE_MODEL_NAME = "hey_jarvis_v0.1"
WAKE_MODEL_FILENAME = f"{WAKE_MODEL_NAME}.onnx"


def required_model_path(package_file: str | None) -> Path:
    if package_file is None:
        raise HarnessError("Could not locate OpenWakeWord package resources")
    return Path(package_file).parent / "resources" / "models" / WAKE_MODEL_FILENAME


def ensure_required_model() -> Path:
    """Download the required wake model only when the installed asset is absent."""

    import openwakeword

    model_path = required_model_path(openwakeword.__file__)
    if model_path.is_file():
        return model_path

    from openwakeword.utils import download_file, download_models

    download_models([WAKE_MODEL_NAME])
    if not model_path.is_file():
        model_urls = [
            details["download_url"]
            for details in openwakeword.MODELS.values()
            if WAKE_MODEL_NAME in details["download_url"]
        ]
        if len(model_urls) != 1:
            raise HarnessError(f"Could not resolve download for {WAKE_MODEL_FILENAME}")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        download_file(
            model_urls[0].replace(".tflite", ".onnx"),
            str(model_path.parent),
        )
    if not model_path.is_file():
        raise HarnessError(
            f"OpenWakeWord did not install required model {WAKE_MODEL_FILENAME}"
        )
    return model_path


def main() -> None:
    print(f"OpenWakeWord model ready: {ensure_required_model()}")


if __name__ == "__main__":
    main()
