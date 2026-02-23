#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path

from alsa_realtime_audio import AlsaPcmDuplex


def _ensure_pcm16_wav(path: str) -> tuple[str, bool]:
    """
    Ensure the WAV can be consumed by the PCM16 playback backend.
    Returns (resolved_path, is_temporary_conversion).
    """
    try:
        with wave.open(path, "rb") as wf:
            if wf.getsampwidth() == 2:
                return path, False
    except wave.Error:
        pass

    sox = shutil.which("sox")
    if sox is None:
        raise RuntimeError(
            "Input WAV is not PCM16 and 'sox' is not installed. "
            "Install sox or provide a PCM16 WAV."
        )

    fd, tmp_path = tempfile.mkstemp(prefix="main_pcm16_", suffix=".wav")
    os.close(fd)
    cmd = [sox, path, "-b", "16", "-e", "signed-integer", "-t", "wav", tmp_path]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        Path(tmp_path).unlink(missing_ok=True)
        stderr = exc.stderr.strip() if exc.stderr else str(exc)
        raise RuntimeError(f"sox conversion failed: {stderr}") from exc
    return tmp_path, True


def main() -> None:
    parser = argparse.ArgumentParser(description="Play a WAV file through speaker output.")
    parser.add_argument("--wav", required=True, help="Path to input WAV file.")
    args = parser.parse_args()

    input_path = Path(args.wav).expanduser()
    if not input_path.exists():
        raise FileNotFoundError(f"WAV file not found: {input_path}")

    wav_path, converted_temp = _ensure_pcm16_wav(str(input_path))
    try:
        with wave.open(wav_path, "rb") as wf:
            channels = wf.getnchannels()
            rate = wf.getframerate()
            width = wf.getsampwidth()
        if width != 2:
            raise ValueError(f"WAV sample width={width} is not PCM16")

        audio = AlsaPcmDuplex(
            device="sysdefault",
            rate=rate,
            channels=channels,
            frames_per_block=256,
        )
        played = audio.play_wav(wav_path)
        print(
            {
                "wav": str(input_path),
                "played_frames": played,
                "device": "sysdefault",
                "channels": channels,
                "rate": rate,
                "converted_to_pcm16": converted_temp,
            }
        )
    finally:
        if converted_temp:
            Path(wav_path).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
