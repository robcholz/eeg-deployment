#!/usr/bin/env python3
from __future__ import annotations

import argparse
import array
import sys
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alsa_realtime_audio import AlsaPcmDuplex


def build_panned_stereo_wav(src: str, dst: str, side: str) -> None:
    with wave.open(src, "rb") as r:
        ch = r.getnchannels()
        sw = r.getsampwidth()
        rate = r.getframerate()
        n = r.getnframes()
        raw = r.readframes(n)
    if sw != 2:
        raise ValueError(f"Unsupported sample width in {src}: {sw}")

    samples = array.array("h")
    samples.frombytes(raw)
    if ch == 1:
        mono = samples.tolist()
    elif ch == 2:
        mono = [(samples[i] + samples[i + 1]) // 2 for i in range(0, len(samples), 2)]
    else:
        raise ValueError(f"Unsupported channel count in {src}: {ch}")

    out = array.array("h")
    for v in mono:
        if side == "left":
            out.append(v)
            out.append(0)
        elif side == "right":
            out.append(0)
            out.append(v)
        else:
            raise ValueError("side must be 'left' or 'right'")

    with wave.open(dst, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(out.tobytes())


def main() -> None:
    parser = argparse.ArgumentParser(description="Play spoken left/right channel prompts.")
    parser.add_argument("--device", default="hw:2,0")
    parser.add_argument("--rate", type=int, default=48_000)
    parser.add_argument("--frames-per-block", type=int, default=256)
    args = parser.parse_args()

    src_left = "/usr/share/sounds/alsa/Front_Left.wav"
    src_right = "/usr/share/sounds/alsa/Front_Right.wav"
    if not Path(src_left).exists() or not Path(src_right).exists():
        raise RuntimeError("Missing ALSA prompt WAVs at /usr/share/sounds/alsa")

    left_wav = "/tmp/front_left_stereo.wav"
    right_wav = "/tmp/front_right_stereo.wav"
    build_panned_stereo_wav(src_left, left_wav, "left")
    build_panned_stereo_wav(src_right, right_wav, "right")

    audio = AlsaPcmDuplex(
        device=args.device,
        rate=args.rate,
        channels=2,
        frames_per_block=args.frames_per_block,
    )
    left_frames = audio.play_wav(left_wav)
    time.sleep(0.5)
    right_frames = audio.play_wav(right_wav)
    print(
        {
            "left_wav": left_wav,
            "right_wav": right_wav,
            "left_frames": left_frames,
            "right_frames": right_frames,
            "device": args.device,
        }
    )


if __name__ == "__main__":
    main()
