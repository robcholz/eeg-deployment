#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alsa_realtime_audio import AlsaPcmDuplex
import sounddevice as sd


def _resolve_device(device: str) -> str | int:
    devices = sd.query_devices()
    if device.isdigit():
        return int(device)
    for idx, dev in enumerate(devices):
        if dev["name"] == device:
            return idx
    if device.startswith("hw:"):
        probe = f"({device})"
        for idx, dev in enumerate(devices):
            if probe in dev["name"] and dev["max_output_channels"] > 0:
                return idx
        print(f"Warning: '{device}' not visible to sounddevice; falling back to 'sysdefault'")
        return "sysdefault"
    return device


def main() -> None:
    parser = argparse.ArgumentParser(description="Play a WAV file using AlsaPcmDuplex API.")
    parser.add_argument("--wav", required=True, help="Path to WAV file.")
    parser.add_argument("--device", default="sysdefault")
    parser.add_argument("--rate", type=int, default=None)
    parser.add_argument("--channels", type=int, default=None)
    parser.add_argument("--frames-per-block", type=int, default=256)
    parser.add_argument(
        "--strict-format",
        action="store_true",
        help="Fail if --rate/--channels do not match WAV metadata.",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List sounddevice output devices and exit.",
    )
    args = parser.parse_args()

    if args.list_devices:
        for i, dev in enumerate(sd.query_devices()):
            if dev["max_output_channels"] > 0:
                print(f"{i}: {dev['name']} (max_out={dev['max_output_channels']})")
        return

    with wave.open(args.wav, "rb") as wf:
        wav_channels = wf.getnchannels()
        wav_rate = wf.getframerate()
        wav_width = wf.getsampwidth()

    if wav_width != 2:
        raise ValueError(f"WAV sample width={wav_width} is not PCM16")

    rate = args.rate if args.rate is not None else wav_rate
    channels = args.channels if args.channels is not None else wav_channels

    if rate != wav_rate or channels != wav_channels:
        if args.strict_format:
            raise ValueError(
                f"WAV format is channels={wav_channels}, rate={wav_rate}, "
                f"but requested channels={channels}, rate={rate}"
            )
        print(
            f"Warning: overriding requested format to WAV metadata "
            f"(channels={wav_channels}, rate={wav_rate})"
        )
        rate = wav_rate
        channels = wav_channels

    device = _resolve_device(args.device)
    audio = AlsaPcmDuplex(
        device=device,
        rate=rate,
        channels=channels,
        frames_per_block=args.frames_per_block,
    )
    played = audio.play_wav(args.wav)
    print(
        {
            "wav": args.wav,
            "played_frames": played,
            "device": device,
            "channels": channels,
            "rate": rate,
        }
    )


if __name__ == "__main__":
    main()
