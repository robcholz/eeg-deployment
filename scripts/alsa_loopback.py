#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alsa_realtime_audio import AlsaPcmDuplex


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-time mic->speaker loopback.")
    parser.add_argument("--device", default="sysdefault")
    parser.add_argument("--rate", type=int, default=48_000)
    parser.add_argument("--channels", type=int, default=2)
    parser.add_argument("--frames-per-block", type=int, default=256)
    parser.add_argument("--queue-frames", type=int, default=64)
    parser.add_argument("--seconds", type=int, default=3)
    args = parser.parse_args()

    audio = AlsaPcmDuplex(
        device=args.device,
        rate=args.rate,
        channels=args.channels,
        frames_per_block=args.frames_per_block,
        queue_frames=args.queue_frames,
    )
    audio.start_duplex()
    stop_at = time.time() + args.seconds
    try:
        while time.time() < stop_at:
            frame = audio.read_mic_frame(timeout_s=0.1)
            if frame is None:
                continue
            audio.write_speaker_frame(frame, timeout_s=0.1)
    finally:
        audio.stop()

    print(asdict(audio.get_stats()))


if __name__ == "__main__":
    main()
