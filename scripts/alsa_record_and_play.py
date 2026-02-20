#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alsa_realtime_audio import AlsaPcmDuplex


def main() -> None:
    parser = argparse.ArgumentParser(description="Record from mic, then play the WAV.")
    parser.add_argument("--device", default="sysdefault")
    parser.add_argument("--rate", type=int, default=48_000)
    parser.add_argument("--channels", type=int, default=2)
    parser.add_argument("--frames-per-block", type=int, default=256)
    parser.add_argument("--seconds", type=int, default=3)
    parser.add_argument("--wav", default="/tmp/alsa_record_and_play.wav")
    args = parser.parse_args()

    audio = AlsaPcmDuplex(
        device=args.device,
        rate=args.rate,
        channels=args.channels,
        frames_per_block=args.frames_per_block,
    )
    recorded = audio.record_to_wav(args.wav, seconds=args.seconds)
    played = audio.play_wav(args.wav)
    print({"wav": args.wav, "recorded_frames": recorded, "played_frames": played})


if __name__ == "__main__":
    main()
