from __future__ import annotations

import queue
import threading
import time
import wave
from dataclasses import dataclass
from typing import Optional

try:
    import sounddevice as sd
except ModuleNotFoundError as exc:
    raise RuntimeError(
        "Missing dependency: sounddevice. Install with: pip install sounddevice"
    ) from exc


@dataclass
class StreamStats:
    frames_in: int = 0
    frames_out: int = 0
    bytes_in: int = 0
    bytes_out: int = 0
    input_overflows: int = 0
    input_underflows: int = 0
    output_overflows: int = 0
    output_underflows: int = 0
    queue_drops: int = 0
    started_at_s: float = 0.0

    @property
    def uptime_s(self) -> float:
        if self.started_at_s <= 0:
            return 0.0
        return max(0.0, time.time() - self.started_at_s)


class AlsaPcmDuplex:
    """
    Minimal real-time local audio API:
    - PCM16 at fixed rate/channels/frames_per_block
    - callback capture + callback playback
    - in-process queues for mic->app and app->speaker
    """

    def __init__(
            self,
            device: str = "sysdefault",
            rate: int = 48_000,
            channels: int = 2,
            frames_per_block: int = 256,
            queue_frames: int = 64,
    ) -> None:
        self.device = device
        self.rate = rate
        self.channels = channels
        self.frames_per_block = frames_per_block
        self.bytes_per_frame = 2 * channels  # PCM16

        self._mic_queue: queue.Queue[bytes] = queue.Queue(maxsize=queue_frames)
        self._spk_queue: queue.Queue[bytes] = queue.Queue(maxsize=queue_frames)
        self._stats = StreamStats()
        self._stats_lock = threading.Lock()
        self._running = False

        self._in_stream: Optional[sd.RawInputStream] = None
        self._out_stream: Optional[sd.RawOutputStream] = None

    def start_input_stream(self) -> None:
        if self._in_stream is not None:
            return
        self._mark_started()
        self._in_stream = sd.RawInputStream(
            samplerate=self.rate,
            channels=self.channels,
            dtype="int16",
            device=self.device,
            blocksize=self.frames_per_block,
            callback=self._on_input,
        )
        self._in_stream.start()
        self._running = True

    def start_output_stream(self) -> None:
        if self._out_stream is not None:
            return
        self._mark_started()
        self._out_stream = sd.RawOutputStream(
            samplerate=self.rate,
            channels=self.channels,
            dtype="int16",
            device=self.device,
            blocksize=self.frames_per_block,
            callback=self._on_output,
        )
        self._out_stream.start()
        self._running = True

    def start_duplex(self) -> None:
        self.start_input_stream()
        self.start_output_stream()

    def stop_input_stream(self) -> None:
        if self._in_stream is None:
            return
        self._in_stream.stop()
        self._in_stream.close()
        self._in_stream = None
        self._running = self._out_stream is not None

    def stop_output_stream(self) -> None:
        if self._out_stream is None:
            return
        self._out_stream.stop()
        self._out_stream.close()
        self._out_stream = None
        self._running = self._in_stream is not None

    def stop(self) -> None:
        self.stop_input_stream()
        self.stop_output_stream()
        self._running = False

    def read_mic_frame(self, timeout_s: Optional[float] = None) -> Optional[bytes]:
        try:
            return self._mic_queue.get(timeout=timeout_s)
        except queue.Empty:
            return None

    def write_speaker_frame(self, frame: bytes, timeout_s: Optional[float] = None) -> bool:
        expected = self.frames_per_block * self.bytes_per_frame
        if len(frame) != expected:
            raise ValueError(f"Expected {expected} bytes, got {len(frame)}")
        try:
            self._spk_queue.put(frame, timeout=timeout_s)
            return True
        except queue.Full:
            with self._stats_lock:
                self._stats.queue_drops += 1
            return False

    def get_stats(self) -> StreamStats:
        with self._stats_lock:
            return StreamStats(
                frames_in=self._stats.frames_in,
                frames_out=self._stats.frames_out,
                bytes_in=self._stats.bytes_in,
                bytes_out=self._stats.bytes_out,
                input_overflows=self._stats.input_overflows,
                input_underflows=self._stats.input_underflows,
                output_overflows=self._stats.output_overflows,
                output_underflows=self._stats.output_underflows,
                queue_drops=self._stats.queue_drops,
                started_at_s=self._stats.started_at_s,
            )

    def record_to_wav(self, path: str, seconds: float, timeout_s: float = 1.0) -> int:
        if seconds <= 0:
            raise ValueError("seconds must be > 0")

        started_here = self._in_stream is None
        if started_here:
            self.start_input_stream()

        target_frames = int(self.rate * seconds)
        target_bytes = target_frames * self.bytes_per_frame
        data = bytearray()
        try:
            while len(data) < target_bytes:
                frame = self.read_mic_frame(timeout_s=timeout_s)
                if frame is None:
                    continue
                data.extend(frame)
        finally:
            if started_here:
                self.stop_input_stream()

        payload = bytes(data[:target_bytes])
        with wave.open(path, "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)
            wf.setframerate(self.rate)
            wf.writeframes(payload)
        return target_frames

    def play_wav(self, path: str, timeout_s: float = 1.0) -> int:
        with wave.open(path, "rb") as wf:
            channels = wf.getnchannels()
            width = wf.getsampwidth()
            rate = wf.getframerate()
            total_frames = wf.getnframes()
            if channels != self.channels:
                raise ValueError(
                    f"WAV channels={channels} does not match stream channels={self.channels}"
                )
            if width != 2:
                raise ValueError(f"WAV sample width={width} is not PCM16")
            if rate != self.rate:
                raise ValueError(f"WAV rate={rate} does not match stream rate={self.rate}")

            started_here = self._out_stream is None
            if started_here:
                self.start_output_stream()
            try:
                block_bytes = self.frames_per_block * self.bytes_per_frame
                while True:
                    chunk = wf.readframes(self.frames_per_block)
                    if not chunk:
                        break
                    if len(chunk) < block_bytes:
                        chunk = chunk + (b"\x00" * (block_bytes - len(chunk)))
                    self.write_speaker_frame(chunk, timeout_s=timeout_s)
                while not self._spk_queue.empty():
                    time.sleep(self.frames_per_block / self.rate)
            finally:
                if started_here:
                    self.stop_output_stream()
        return total_frames

    def _mark_started(self) -> None:
        with self._stats_lock:
            if self._stats.started_at_s == 0.0:
                self._stats.started_at_s = time.time()

    def _on_input(self, indata, frames, _time_info, status) -> None:
        data = bytes(indata)
        with self._stats_lock:
            self._stats.frames_in += frames
            self._stats.bytes_in += len(data)
            if status.input_overflow:
                self._stats.input_overflows += 1
            if status.input_underflow:
                self._stats.input_underflows += 1
        try:
            self._mic_queue.put_nowait(data)
        except queue.Full:
            with self._stats_lock:
                self._stats.queue_drops += 1

    def _on_output(self, outdata, frames, _time_info, status) -> None:
        expected = frames * self.bytes_per_frame
        try:
            data = self._spk_queue.get_nowait()
        except queue.Empty:
            data = b"\x00" * expected

        if len(data) != expected:
            data = (data + (b"\x00" * expected))[:expected]

        outdata[:] = data
        with self._stats_lock:
            self._stats.frames_out += frames
            self._stats.bytes_out += expected
            if status.output_overflow:
                self._stats.output_overflows += 1
            if status.output_underflow:
                self._stats.output_underflows += 1
