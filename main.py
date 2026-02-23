#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import time
import wave
from dataclasses import asdict, dataclass, field
from pathlib import Path

from alsa_realtime_audio import AlsaPcmDuplex

SNN_SOUND_DIR = Path(__file__).resolve().parent / "SNN_Sound"
SNN_MODEL_PATH = SNN_SOUND_DIR / "save_models" / "best_snn.pt"
LOGGER = logging.getLogger("eeg_deployment.main")

STREAM_CHUNK_SECONDS = 0.5
STREAM_QUEUE_TIMEOUT_S = 1.0
STREAM_PROGRESS_EVERY_CHUNKS = 4
STREAM_PREBUFFER_SECONDS = 0.2


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
    gains: list[float] = field(default_factory=list)
    mixed_samples: int | None = None
    playback_device: str = "sysdefault"
    playback_channels: int = 1
    played_frames: int | None = None
    chunks_total: int | None = None
    chunks_processed: int = 0
    queue_drops: int = 0
    enqueued_frames: int = 0
    enqueued_blocks: int = 0
    dropped_blocks: int = 0
    drained: bool | None = None
    consumer_consumed_all: bool | None = None
    output_underflows: int | None = None
    prebuffer_blocks: int | None = None
    inserted_silence_frames: int = 0
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
        f"chunks={state.chunks_processed}/{state.chunks_total}",
    ]
    if state.consumer_consumed_all is not None:
        fields.append(f"consumer_consumed_all={state.consumer_consumed_all}")
    if state.drained is not None:
        fields.append(f"drained={state.drained}")
    if state.output_underflows is not None:
        fields.append(f"output_underflows={state.output_underflows}")
    if state.prebuffer_blocks is not None:
        fields.append(f"prebuffer_blocks={state.prebuffer_blocks}")
    if state.inserted_silence_frames:
        fields.append(f"inserted_silence_frames={state.inserted_silence_frames}")
    if state.gains:
        fields.append(f"gains={_short(state.gains)}")
    if state.timings_s:
        fields.append(f"timings_s={_short(state.timings_s)}")
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


def _read_wav_header(path: Path) -> dict[str, int]:
    with wave.open(str(path), "rb") as wf:
        if wf.getcomptype() != "NONE":
            raise ValueError(f"Compressed WAV is not supported: comptype={wf.getcomptype()}")
        return {
            "sample_rate": wf.getframerate(),
            "channels": wf.getnchannels(),
            "sample_width": wf.getsampwidth(),
            "frames": wf.getnframes(),
        }


def _pcm_bytes_to_mono_float(raw_bytes: bytes, sample_width: int, channels: int) -> np.ndarray:
    import numpy as np

    if sample_width == 1:
        audio = np.frombuffer(raw_bytes, dtype=np.uint8)
    elif sample_width == 2:
        audio = np.frombuffer(raw_bytes, dtype=np.int16)
    elif sample_width == 4:
        audio = np.frombuffer(raw_bytes, dtype=np.int32)
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width}")

    if channels > 1:
        valid = audio.size - (audio.size % channels)
        audio = audio[:valid]
        if audio.size == 0:
            return np.zeros(0, dtype=np.float32)
        audio = audio.reshape(-1, channels).mean(axis=1)

    return _to_float32(audio).astype(np.float32)


def _import_snn_modules() -> tuple[type, object]:
    if str(SNN_SOUND_DIR) not in sys.path:
        sys.path.insert(0, str(SNN_SOUND_DIR))
    from snn_model import SimpleSNN
    from spikingjelly.clock_driven import functional

    return SimpleSNN, functional


class SnnSeparator:
    def __init__(self) -> None:
        import torch

        if not SNN_MODEL_PATH.exists():
            raise FileNotFoundError(f"SNN model checkpoint not found: {SNN_MODEL_PATH}")

        self._torch = torch
        self._n_fft = 512
        self._hop_length = 160

        simple_snn, functional = _import_snn_modules()
        self._functional = functional
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = simple_snn(input_size=257, hidden_size=512, T=6).to(self.device)

        state_dict = torch.load(str(SNN_MODEL_PATH), map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.window = torch.hann_window(self._n_fft).to(self.device)

    def separate_chunk(self, audio_mono: np.ndarray) -> list[np.ndarray]:
        import numpy as np

        torch = self._torch
        target_len = int(audio_mono.shape[0])
        if target_len == 0:
            empty = np.zeros(0, dtype=np.float32)
            return [empty, empty]

        chunk = audio_mono.astype(np.float32, copy=False)
        if target_len < self._n_fft:
            chunk = np.pad(chunk, (0, self._n_fft - target_len))

        mix_tensor = torch.from_numpy(chunk).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            mix_stft = torch.stft(
                mix_tensor,
                n_fft=self._n_fft,
                hop_length=self._hop_length,
                return_complex=True,
                window=self.window,
            )
            mix_mag = torch.abs(mix_stft)
            mix_phase = torch.angle(mix_stft)
            mix_input = torch.log1p(mix_mag)

            self._functional.reset_net(self.model)
            m1_pred, m2_pred = self.model(mix_input)

            est_mag1 = mix_mag * m1_pred
            est_mag2 = mix_mag * m2_pred

            est_s1 = torch.istft(
                est_mag1 * torch.exp(1j * mix_phase),
                n_fft=self._n_fft,
                hop_length=self._hop_length,
                window=self.window,
            )
            est_s2 = torch.istft(
                est_mag2 * torch.exp(1j * mix_phase),
                n_fft=self._n_fft,
                hop_length=self._hop_length,
                window=self.window,
            )

        source_1 = est_s1.squeeze().cpu().numpy().astype(np.float32)
        source_2 = est_s2.squeeze().cpu().numpy().astype(np.float32)
        source_1 = source_1[:target_len]
        source_2 = source_2[:target_len]
        if source_1.shape[0] < target_len:
            source_1 = np.pad(source_1, (0, target_len - source_1.shape[0]))
        if source_2.shape[0] < target_len:
            source_2 = np.pad(source_2, (0, target_len - source_2.shape[0]))
        return [source_1, source_2]


def apply_source_gains(sources: list[np.ndarray], gains: list[float]) -> np.ndarray:
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


class _SpeakerBlockPacker:
    """
    Convert variable-size mono chunks to fixed-size speaker blocks continuously.
    Important: do not pad per chunk; pad once at final flush only.
    """

    def __init__(self, audio: AlsaPcmDuplex) -> None:
        self.audio = audio
        self.source_frames = 0
        self.enqueued_blocks = 0
        self.dropped_blocks = 0
        self.inserted_silence_frames = 0
        self._buffer = bytearray()
        self._offset = 0

    def _available_bytes(self) -> int:
        return len(self._buffer) - self._offset

    def push(self, mono_audio: np.ndarray, timeout_s: float, on_block_enqueued=None) -> None:
        import numpy as np

        pcm16 = (np.clip(mono_audio, -1.0, 1.0) * 32767.0).astype(np.int16)
        self.source_frames += int(pcm16.shape[0])
        if self._offset == len(self._buffer):
            self._buffer.clear()
            self._offset = 0
        self._buffer.extend(pcm16.tobytes())
        self._flush_full_blocks(timeout_s=timeout_s, on_block_enqueued=on_block_enqueued)

    def flush(self, timeout_s: float, on_block_enqueued=None) -> None:
        block_bytes = self.audio.frames_per_block * self.audio.bytes_per_frame
        available = self._available_bytes()
        if available > 0:
            pad = (-available) % block_bytes
            if pad > 0:
                self._buffer.extend(b"\x00" * pad)
                self.inserted_silence_frames += pad // self.audio.bytes_per_frame
        self._flush_full_blocks(timeout_s=timeout_s, on_block_enqueued=on_block_enqueued)

    def _flush_full_blocks(self, timeout_s: float, on_block_enqueued=None) -> None:
        block_bytes = self.audio.frames_per_block * self.audio.bytes_per_frame
        while self._available_bytes() >= block_bytes:
            start = self._offset
            end = start + block_bytes
            block = bytes(self._buffer[start:end])
            self._offset = end
            ok = self.audio.write_speaker_frame(block, timeout_s=timeout_s)
            if ok:
                self.enqueued_blocks += 1
                if on_block_enqueued is not None:
                    on_block_enqueued()
            else:
                self.dropped_blocks += 1

        if self._offset > 0 and (self._offset == len(self._buffer) or self._offset > 262144):
            del self._buffer[:self._offset]
            self._offset = 0


def _wait_for_playback_drain(audio: AlsaPcmDuplex, timeout_s: float) -> bool:
    deadline = time.time() + timeout_s
    sleep_s = audio.frames_per_block / audio.rate
    # Reuse the same queue-drain condition used by AlsaPcmDuplex.play_wav().
    while time.time() < deadline:
        if audio._spk_queue.empty():
            return True
        time.sleep(sleep_s)
    return audio._spk_queue.empty()


def _stream_process_and_play(input_path: Path, state: ProgramState) -> None:
    header = _read_wav_header(input_path)
    state.sample_rate = int(header["sample_rate"])
    state.input_samples = int(header["frames"])

    chunk_samples = max(1, int(state.sample_rate * STREAM_CHUNK_SECONDS))
    state.chunks_total = int(math.ceil(state.input_samples / chunk_samples))
    _log_event(
        "wav_header",
        state,
        channels=header["channels"],
        sample_width=header["sample_width"],
        chunk_samples=chunk_samples,
    )

    t0 = time.perf_counter()
    separator = SnnSeparator()
    state.timings_s["model_load"] = round(time.perf_counter() - t0, 4)
    _log_event(
        "snn_model_loaded",
        state,
        model_device=str(separator.device),
        checkpoint=str(SNN_MODEL_PATH),
    )

    audio = AlsaPcmDuplex(
        device=state.playback_device,
        rate=state.sample_rate,
        channels=state.playback_channels,
        frames_per_block=256,
    )
    packer = _SpeakerBlockPacker(audio=audio)

    separation_s = 0.0
    gain_mix_s = 0.0
    enqueue_s = 0.0
    stream_started = False
    prebuffer_blocks = max(
        4,
        min(
            32,
            int((STREAM_PREBUFFER_SECONDS * state.sample_rate) / audio.frames_per_block),
        ),
    )
    state.prebuffer_blocks = prebuffer_blocks

    state.phase = "stream_playback"

    def _maybe_start_output_stream() -> None:
        nonlocal stream_started
        if stream_started:
            return
        if packer.enqueued_blocks < prebuffer_blocks:
            return
        audio.start_output_stream()
        stream_started = True
        _log_event(
            "output_stream_started",
            state,
            prebuffer_seconds=round(
                (prebuffer_blocks * audio.frames_per_block) / state.sample_rate, 4
            ),
        )

    try:
        with wave.open(str(input_path), "rb") as wf:
            chunk_idx = 0
            while True:
                raw = wf.readframes(chunk_samples)
                if not raw:
                    break
                chunk_idx += 1

                chunk = _pcm_bytes_to_mono_float(
                    raw_bytes=raw,
                    sample_width=int(header["sample_width"]),
                    channels=int(header["channels"]),
                )
                if chunk.size == 0:
                    continue

                t_sep = time.perf_counter()
                separated_sources = separator.separate_chunk(chunk)
                separation_s += time.perf_counter() - t_sep

                if state.num_sources is None:
                    state.num_sources = len(separated_sources)
                    state.gains = [1.0 for _ in separated_sources] # todo
                    _log_event("gains_initialized", state)

                t_mix = time.perf_counter()
                mixed_output = apply_source_gains(separated_sources, state.gains)
                gain_mix_s += time.perf_counter() - t_mix

                t_enq = time.perf_counter()
                packer.push(
                    mono_audio=mixed_output,
                    timeout_s=STREAM_QUEUE_TIMEOUT_S,
                    on_block_enqueued=_maybe_start_output_stream,
                )
                enqueue_s += time.perf_counter() - t_enq

                state.enqueued_frames = int(packer.source_frames)
                state.enqueued_blocks = int(packer.enqueued_blocks)
                state.dropped_blocks = int(packer.dropped_blocks)
                state.inserted_silence_frames = int(packer.inserted_silence_frames)
                state.chunks_processed = chunk_idx
                state.mixed_samples = int(packer.source_frames)

                if (
                        chunk_idx == 1
                        or chunk_idx == state.chunks_total
                        or chunk_idx % STREAM_PROGRESS_EVERY_CHUNKS == 0
                ):
                    _log_event(
                        "stream_progress",
                        state,
                        chunk_audio=_audio_array_stats(mixed_output),
                        stream_started=stream_started,
                        queue_size=audio._spk_queue.qsize(),
                        playback_stats=asdict(audio.get_stats()),
                    )

        t_enq = time.perf_counter()
        packer.flush(
            timeout_s=STREAM_QUEUE_TIMEOUT_S,
            on_block_enqueued=_maybe_start_output_stream,
        )
        enqueue_s += time.perf_counter() - t_enq
        state.enqueued_frames = int(packer.source_frames)
        state.enqueued_blocks = int(packer.enqueued_blocks)
        state.dropped_blocks = int(packer.dropped_blocks)
        state.inserted_silence_frames = int(packer.inserted_silence_frames)
        state.mixed_samples = int(packer.source_frames)
        _log_event("speaker_flush_done", state, stream_started=stream_started)

        if not stream_started:
            audio.start_output_stream()
            stream_started = True
            _log_event(
                "output_stream_started",
                state,
                reason="late_start_short_audio",
            )

        state.phase = "stream_drain"
        _log_event(
            "drain_start",
            state,
            enqueued_frames=state.enqueued_frames,
            enqueued_blocks=state.enqueued_blocks,
            dropped_blocks=state.dropped_blocks,
        )
        enqueued_playback_frames = state.enqueued_blocks * audio.frames_per_block
        drain_timeout_s = max(3.0, (enqueued_playback_frames / state.sample_rate) + 3.0)
        drained = _wait_for_playback_drain(audio, timeout_s=drain_timeout_s)
        stats = audio.get_stats()
        state.played_frames = int(stats.frames_out)
        state.queue_drops = int(stats.queue_drops)
        state.drained = bool(drained)
        state.output_underflows = int(stats.output_underflows)
        state.consumer_consumed_all = bool(
            drained and state.dropped_blocks == 0 and state.queue_drops == 0
        )
        _log_event("drain_done", state, drained=drained, playback_stats=asdict(stats))
        _log_event(
            "consumption_check",
            state,
            enqueued_frames=state.enqueued_frames,
            enqueued_blocks=state.enqueued_blocks,
            dropped_blocks=state.dropped_blocks,
            queue_drops=state.queue_drops,
            inserted_silence_frames=state.inserted_silence_frames,
        )
        if not state.consumer_consumed_all:
            LOGGER.warning(
                "consumer did not fully drain produced audio blocks "
                "(drained=%s dropped_blocks=%s)",
                drained,
                state.dropped_blocks,
            )
    finally:
        if stream_started:
            audio.stop_output_stream()
            _log_event("output_stream_stopped", state)

    state.timings_s["snn_separation"] = round(separation_s, 4)
    state.timings_s["apply_gains"] = round(gain_mix_s, 4)
    state.timings_s["enqueue_speaker"] = round(enqueue_s, 4)


def main() -> None:
    _configure_logging()
    parser = argparse.ArgumentParser(
        description="Stream input through SNN_Sound separation and play while processing."
    )
    parser.add_argument("--wav", required=True, help="Path to input WAV file.")
    args = parser.parse_args()

    total_t0 = time.perf_counter()
    input_path = Path(args.wav).expanduser()
    state = ProgramState(wav=str(input_path))
    _log_event(
        "program_start",
        state,
        pid=os.getpid(),
        python=sys.version.split()[0],
        checkpoint=str(SNN_MODEL_PATH),
        stream_chunk_seconds=STREAM_CHUNK_SECONDS,
        stream_prebuffer_seconds=STREAM_PREBUFFER_SECONDS,
    )

    try:
        state.phase = "validate_input"
        _log_event("input_validation_start", state)
        if not input_path.exists():
            raise FileNotFoundError(f"WAV file not found: {input_path}")

        _stream_process_and_play(input_path=input_path, state=state)

        state.phase = "completed"
        state.timings_s["total"] = round(time.perf_counter() - total_t0, 4)
        _log_event("program_done", state)
        LOGGER.info("summary {}".format(
                   {
                       "wav": state.wav,
                       "played_frames": state.played_frames,
                       "device": state.playback_device,
                       "channels": state.playback_channels,
                       "rate": state.sample_rate,
                       "num_sources": state.num_sources,
                       "gains": state.gains,
                       "chunks_processed": state.chunks_processed,
                       "chunks_total": state.chunks_total,
                       "queue_drops": state.queue_drops,
                       "enqueued_frames": state.enqueued_frames,
                       "enqueued_blocks": state.enqueued_blocks,
                       "dropped_blocks": state.dropped_blocks,
                       "inserted_silence_frames": state.inserted_silence_frames,
                       "prebuffer_blocks": state.prebuffer_blocks,
                       "drained": state.drained,
                       "output_underflows": state.output_underflows,
                       "consumer_consumed_all": state.consumer_consumed_all,
                       "timings_s": state.timings_s,
                   })
                   )
    except Exception as exc:
        state.phase = "error"
        state.timings_s["total"] = round(time.perf_counter() - total_t0, 4)
        _log_event("program_error", state, error=str(exc))
        LOGGER.exception("program_error_traceback")
        raise


if __name__ == "__main__":
    main()
