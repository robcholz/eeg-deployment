#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from alsa_realtime_audio import AlsaPcmDuplex

SNN_SOUND_DIR = Path(__file__).resolve().parent / "SNN_Sound"
SNN_MODEL_PATH = SNN_SOUND_DIR / "save_models" / "best_snn.pt"
LOGGER = logging.getLogger("eeg_deployment.main")


class ColorFormatter(logging.Formatter):
    RESET = "\x1b[0m"
    COLORS = {
        logging.DEBUG: "\x1b[36m",
        logging.INFO: "\x1b[32m",
        logging.WARNING: "\x1b[33m",
        logging.ERROR: "\x1b[31m",
        logging.CRITICAL: "\x1b[35m",
    }

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelno, "")
        if not color:
            return super().format(record)

        original_levelname = record.levelname
        record.levelname = f"{color}{original_levelname}{self.RESET}"
        try:
            return super().format(record)
        finally:
            record.levelname = original_levelname


@dataclass
class ProgramState:
    phase: str = "init"
    wav: str = ""
    sample_rate: int | None = None
    input_samples: int | None = None
    num_sources: int | None = None
    source_lengths: list[int] = field(default_factory=list)
    gains: list[float] = field(default_factory=list)
    mixed_samples: int | None = None
    temp_wav: str = ""
    playback_device: str = "sysdefault"
    playback_channels: int = 1
    played_frames: int | None = None
    timings_s: dict[str, float] = field(default_factory=dict)


def _configure_logging() -> None:
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()

    handler = logging.StreamHandler()
    fmt = "%(asctime)s %(levelname)s %(message)s"
    if hasattr(handler.stream, "isatty") and handler.stream.isatty():
        handler.setFormatter(ColorFormatter(fmt))
    else:
        handler.setFormatter(logging.Formatter(fmt))
    root.addHandler(handler)


def _short(value: object, limit: int = 220) -> str:
    text = repr(value)
    if len(text) > limit:
        return text[: limit - 3] + "..."
    return text


def _log_event(event: str, state: ProgramState, **extra: object) -> None:
    fields = [
        f"event={event}",
        f"phase={state.phase}",
        f"wav={state.wav}",
        f"rate={state.sample_rate}",
        f"input_samples={state.input_samples}",
        f"num_sources={state.num_sources}",
        f"mixed_samples={state.mixed_samples}",
        f"played_frames={state.played_frames}",
        f"device={state.playback_device}",
    ]
    if state.gains:
        fields.append(f"gains={_short(state.gains)}")
    if state.timings_s:
        fields.append(f"timings_s={_short(state.timings_s)}")
    if state.temp_wav:
        fields.append(f"temp_wav={state.temp_wav}")
    for key, value in extra.items():
        fields.append(f"{key}={_short(value)}")
    LOGGER.info(" | ".join(fields))

def _to_float32(audio: np.ndarray) -> np.ndarray:
    import numpy as np

    if audio.dtype == np.float32:
        return np.clip(audio, -1.0, 1.0)
    if audio.dtype == np.float64:
        return np.clip(audio.astype(np.float32), -1.0, 1.0)
    if audio.dtype == np.int16:
        return audio.astype(np.float32) / 32768.0
    if audio.dtype == np.int32:
        return audio.astype(np.float32) / 2147483648.0
    if audio.dtype == np.uint8:
        return (audio.astype(np.float32) - 128.0) / 128.0
    raise ValueError(f"Unsupported WAV dtype: {audio.dtype}")


def _load_mono_wav(path: Path) -> tuple[int, np.ndarray]:
    import numpy as np
    from scipy.io import wavfile

    sample_rate, audio = wavfile.read(str(path))
    audio_f32 = _to_float32(audio)
    if audio_f32.ndim == 2:
        audio_f32 = audio_f32.mean(axis=1)
    if audio_f32.ndim != 1:
        raise ValueError(f"Unsupported WAV shape: {audio_f32.shape}")
    return sample_rate, audio_f32.astype(np.float32)


def _import_snn_modules() -> tuple[type, object]:
    if str(SNN_SOUND_DIR) not in sys.path:
        sys.path.insert(0, str(SNN_SOUND_DIR))
    from snn_model import SimpleSNN
    from spikingjelly.clock_driven import functional

    return SimpleSNN, functional


def _separate_sources_with_snn(audio_mono: np.ndarray) -> list[np.ndarray]:
    import numpy as np
    import torch

    if not SNN_MODEL_PATH.exists():
        raise FileNotFoundError(f"SNN model checkpoint not found: {SNN_MODEL_PATH}")

    SimpleSNN, functional = _import_snn_modules()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleSNN(input_size=257, hidden_size=512, T=6).to(device)

    state_dict = torch.load(str(SNN_MODEL_PATH), map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    mix_tensor = torch.from_numpy(audio_mono).float().unsqueeze(0).to(device)
    n_fft = 512
    hop_length = 160
    window = torch.hann_window(n_fft).to(device)

    with torch.no_grad():
        mix_stft = torch.stft(
            mix_tensor,
            n_fft=n_fft,
            hop_length=hop_length,
            return_complex=True,
            window=window,
        )
        mix_mag = torch.abs(mix_stft)
        mix_phase = torch.angle(mix_stft)
        mix_input = torch.log1p(mix_mag)

        functional.reset_net(model)
        m1_pred, m2_pred = model(mix_input)

        est_mag1 = mix_mag * m1_pred
        est_mag2 = mix_mag * m2_pred

        est_s1 = torch.istft(
            est_mag1 * torch.exp(1j * mix_phase),
            n_fft=n_fft,
            hop_length=hop_length,
            window=window,
        )
        est_s2 = torch.istft(
            est_mag2 * torch.exp(1j * mix_phase),
            n_fft=n_fft,
            hop_length=hop_length,
            window=window,
        )

    source_1 = est_s1.squeeze().cpu().numpy().astype(np.float32)
    source_2 = est_s2.squeeze().cpu().numpy().astype(np.float32)
    target_len = audio_mono.shape[0]
    source_1 = source_1[:target_len]
    source_2 = source_2[:target_len]
    if source_1.shape[0] < target_len:
        source_1 = np.pad(source_1, (0, target_len - source_1.shape[0]))
    if source_2.shape[0] < target_len:
        source_2 = np.pad(source_2, (0, target_len - source_2.shape[0]))

    return [source_1, source_2]


def apply_source_gains(sources: list[np.ndarray], gains: list[float]) -> np.ndarray:
    """
    Minimal gain-array API:
    - sources[i] is multiplied by gains[i]
    - outputs one mixed mono track
    """
    import numpy as np

    if len(sources) == 0:
        raise ValueError("sources must not be empty")
    if len(sources) != len(gains):
        raise ValueError(
            f"gains length ({len(gains)}) must match number of sources ({len(sources)})"
        )

    target_len = max(source.shape[0] for source in sources)
    mixed = np.zeros(target_len, dtype=np.float32)
    for source, gain in zip(sources, gains):
        s = source.astype(np.float32)
        if s.shape[0] < target_len:
            s = np.pad(s, (0, target_len - s.shape[0]))
        mixed += s * float(gain)
    return np.clip(mixed, -1.0, 1.0)


def _audio_array_stats(audio: np.ndarray) -> dict[str, float]:
    import numpy as np

    return {
        "min": float(np.min(audio)),
        "max": float(np.max(audio)),
        "mean": float(np.mean(audio)),
        "rms": float(np.sqrt(np.mean(np.square(audio)))),
    }


def _write_temp_pcm16_wav(sample_rate: int, mono_audio: np.ndarray) -> str:
    import numpy as np
    from scipy.io import wavfile

    pcm16 = (np.clip(mono_audio, -1.0, 1.0) * 32767.0).astype(np.int16)
    fd, temp_path = tempfile.mkstemp(prefix="main_snn_mix_", suffix=".wav")
    os.close(fd)
    wavfile.write(temp_path, sample_rate, pcm16)
    return temp_path


def main() -> None:
    _configure_logging()
    parser = argparse.ArgumentParser(
        description="Separate input with SNN_Sound, apply gains, and play to speaker."
    )
    parser.add_argument("--wav", required=True, help="Path to input WAV file.")
    args = parser.parse_args()

    total_t0 = time.perf_counter()
    input_path = Path(args.wav).expanduser()
    state = ProgramState(wav=str(input_path))
    temp_wav_path = ""
    _log_event(
        "program_start",
        state,
        pid=os.getpid(),
        python=sys.version.split()[0],
        checkpoint=str(SNN_MODEL_PATH),
    )
    try:
        state.phase = "validate_input"
        _log_event("input_validation_start", state)
        if not input_path.exists():
            raise FileNotFoundError(f"WAV file not found: {input_path}")

        step_t0 = time.perf_counter()
        state.phase = "load_wav"
        sample_rate, audio_mono = _load_mono_wav(input_path)
        state.sample_rate = sample_rate
        state.input_samples = int(audio_mono.shape[0])
        state.timings_s["load_wav"] = round(time.perf_counter() - step_t0, 4)
        _log_event("wav_loaded", state, input_audio=_audio_array_stats(audio_mono))

        step_t0 = time.perf_counter()
        state.phase = "snn_separation"
        separated_sources = _separate_sources_with_snn(audio_mono)
        state.num_sources = len(separated_sources)
        state.source_lengths = [int(source.shape[0]) for source in separated_sources]
        state.timings_s["snn_separation"] = round(time.perf_counter() - step_t0, 4)
        _log_event(
            "sources_separated",
            state,
            source_stats=[_audio_array_stats(source) for source in separated_sources],
        )

        step_t0 = time.perf_counter()
        state.phase = "apply_gains"
        gains = [1.0 for _ in separated_sources]
        mixed_output = apply_source_gains(separated_sources, gains)
        state.gains = gains
        state.mixed_samples = int(mixed_output.shape[0])
        state.timings_s["apply_gains"] = round(time.perf_counter() - step_t0, 4)
        _log_event("gains_applied", state, mixed_audio=_audio_array_stats(mixed_output))

        step_t0 = time.perf_counter()
        state.phase = "write_temp_wav"
        temp_wav_path = _write_temp_pcm16_wav(sample_rate, mixed_output)
        state.temp_wav = temp_wav_path
        state.timings_s["write_temp_wav"] = round(time.perf_counter() - step_t0, 4)
        _log_event("temp_wav_ready", state)

        step_t0 = time.perf_counter()
        state.phase = "playback"
        _log_event("playback_start", state)
        audio = AlsaPcmDuplex(
            device=state.playback_device,
            rate=state.sample_rate,
            channels=state.playback_channels,
            frames_per_block=256,
        )
        played = audio.play_wav(temp_wav_path)
        state.played_frames = int(played)
        state.timings_s["playback"] = round(time.perf_counter() - step_t0, 4)
        _log_event("playback_done", state, playback_stats=asdict(audio.get_stats()))

        state.phase = "completed"
        state.timings_s["total"] = round(time.perf_counter() - total_t0, 4)
        _log_event("program_done", state)
        print(
            {
                "wav": state.wav,
                "played_frames": state.played_frames,
                "device": state.playback_device,
                "channels": state.playback_channels,
                "rate": state.sample_rate,
                "num_sources": state.num_sources,
                "gains": state.gains,
                "timings_s": state.timings_s,
            }
        )
    except Exception as exc:
        state.phase = "error"
        state.timings_s["total"] = round(time.perf_counter() - total_t0, 4)
        _log_event("program_error", state, error=str(exc))
        LOGGER.exception("program_error_traceback")
        raise
    finally:
        if temp_wav_path:
            Path(temp_wav_path).unlink(missing_ok=True)
            _log_event("temp_wav_deleted", state, deleted=True)


if __name__ == "__main__":
    main()
