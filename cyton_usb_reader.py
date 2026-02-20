#!/usr/bin/env python3
import argparse
import json
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

try:
    import serial
except ModuleNotFoundError:
    print("Missing dependency: pyserial. Install with: pip install pyserial", file=sys.stderr)
    sys.exit(2)

START_BYTE = 0xA0
PACKET_SIZE = 33
STOP_HIGH_NIBBLE = 0xC0

# ADS1299 conversion from counts to microvolts used by Cyton docs/SDK.
EEG_SCALE_UV = (4.5 / 24.0 / (2 ** 23 - 1)) * 1_000_000.0


@dataclass
class CytonPacket:
    sample_id: int
    eeg_counts: List[int]
    eeg_uV: List[float]
    stop_byte: int
    aux: Dict[str, object]


def int24_to_int32(b0: int, b1: int, b2: int) -> int:
    value = (b0 << 16) | (b1 << 8) | b2
    if value & 0x800000:
        value -= 1 << 24
    return value


def int16_to_int32(b0: int, b1: int) -> int:
    value = (b0 << 8) | b1
    if value & 0x8000:
        value -= 1 << 16
    return value


def parse_aux(stop_byte: int, aux_bytes: bytes) -> Dict[str, object]:
    low_nibble = stop_byte & 0x0F
    aux: Dict[str, object]

    if low_nibble == 0x0:
        aux = {
            "type": "accel_standard",
            "accel_counts": [
                int16_to_int32(aux_bytes[0], aux_bytes[1]),
                int16_to_int32(aux_bytes[2], aux_bytes[3]),
                int16_to_int32(aux_bytes[4], aux_bytes[5]),
            ],
        }
    elif low_nibble == 0x1:
        aux = {"type": "standard_raw_aux", "bytes_hex": aux_bytes.hex()}
    elif low_nibble == 0x2:
        aux = {"type": "user_defined", "bytes_hex": aux_bytes.hex()}
    elif low_nibble in (0x3, 0x4):
        aux = {
            "type": "accel_time_synced",
            "accel_code": chr(aux_bytes[0]),
            "accel_value_byte": aux_bytes[1],
            "board_time": int.from_bytes(aux_bytes[2:6], byteorder="big", signed=False),
        }
    elif low_nibble in (0x5, 0x6):
        aux = {
            "type": "user_defined_time_synced",
            "user_bytes_hex": aux_bytes[:2].hex(),
            "board_time": int.from_bytes(aux_bytes[2:6], byteorder="big", signed=False),
        }
    else:
        aux = {"type": "raw_aux", "bytes_hex": aux_bytes.hex()}

    return aux


def parse_packet(packet: bytes) -> CytonPacket:
    if len(packet) != PACKET_SIZE:
        raise ValueError(f"Expected {PACKET_SIZE} bytes, got {len(packet)}")
    if packet[0] != START_BYTE:
        raise ValueError(f"Invalid start byte: 0x{packet[0]:02X}")
    stop_byte = packet[32]
    if (stop_byte & 0xF0) != STOP_HIGH_NIBBLE:
        raise ValueError(f"Invalid stop byte: 0x{stop_byte:02X}")

    sample_id = packet[1]
    eeg_counts: List[int] = []
    eeg_uV: List[float] = []
    for ch in range(8):
        base = 2 + ch * 3
        count = int24_to_int32(packet[base], packet[base + 1], packet[base + 2])
        eeg_counts.append(count)
        eeg_uV.append(count * EEG_SCALE_UV)

    aux = parse_aux(stop_byte, packet[26:32])
    return CytonPacket(
        sample_id=sample_id,
        eeg_counts=eeg_counts,
        eeg_uV=eeg_uV,
        stop_byte=stop_byte,
        aux=aux,
    )


def extract_packets(buffer: bytearray) -> List[CytonPacket]:
    packets: List[CytonPacket] = []
    while len(buffer) >= PACKET_SIZE:
        start = buffer.find(bytes([START_BYTE]))
        if start < 0:
            buffer.clear()
            break
        if start > 0:
            del buffer[:start]
            if len(buffer) < PACKET_SIZE:
                break

        candidate = bytes(buffer[:PACKET_SIZE])
        if (candidate[32] & 0xF0) == STOP_HIGH_NIBBLE:
            try:
                packets.append(parse_packet(candidate))
                del buffer[:PACKET_SIZE]
                continue
            except ValueError:
                pass

        del buffer[0]
    return packets


def probe_connection(ser: serial.Serial, timeout_s: float = 2.0) -> str:
    ser.reset_input_buffer()
    ser.reset_output_buffer()
    ser.write(b"s")
    time.sleep(0.15)
    ser.reset_input_buffer()
    ser.write(b"v")
    ser.flush()

    deadline = time.time() + timeout_s
    chunks: List[bytes] = []
    while time.time() < deadline:
        data = ser.read(256)
        if data:
            chunks.append(data)
            if b"$$$" in data:
                break
    response = b"".join(chunks).decode(errors="replace")
    return response.strip()


def set_channel_mode(ser: serial.Serial, channels: int) -> str:
    if channels == 16:
        ser.reset_input_buffer()
        ser.write(b"C")
        ser.flush()
        time.sleep(0.25)
        return ser.read(512).decode(errors="replace").strip()
    if channels == 8:
        ser.reset_input_buffer()
        ser.write(b"c")
        ser.flush()
        time.sleep(0.25)
        return ser.read(512).decode(errors="replace").strip()
    raise ValueError("channels must be 8 or 16")


def _iter_cyton_records(
        ser: serial.Serial,
        channels: int,
        duration_s: Optional[float],
        max_samples: Optional[int],
) -> "Iterator[Dict[str, object]]":
    buffer = bytearray()
    sample_count = 0
    start_t = time.time()
    pending_board_pkt: Optional[CytonPacket] = None

    ser.reset_input_buffer()
    ser.write(b"b")
    ser.flush()

    try:
        while True:
            if duration_s is not None and (time.time() - start_t) >= duration_s:
                break
            if max_samples is not None and sample_count >= max_samples:
                break

            chunk = ser.read(1024)
            if not chunk:
                continue
            buffer.extend(chunk)
            parsed = extract_packets(buffer)
            for pkt in parsed:
                if channels == 8:
                    record = {
                        "sample_id": pkt.sample_id,
                        "eeg_counts": pkt.eeg_counts,
                        "eeg_uV": [round(v, 6) for v in pkt.eeg_uV],
                        "stop_byte": f"0x{pkt.stop_byte:02X}",
                        "aux": pkt.aux,
                    }
                    sample_count += 1
                else:
                    # Cyton+Daisy alternates packets:
                    # odd sample numbers -> channels 1-8 (board)
                    # even sample numbers -> channels 9-16 (daisy)
                    # We emit one combined 16-channel sample at 125 Hz.
                    if pkt.sample_id % 2 == 1:
                        pending_board_pkt = pkt
                        continue
                    if pending_board_pkt is None:
                        continue
                    record = {
                        "sample_id": pkt.sample_id,
                        "eeg_counts": pending_board_pkt.eeg_counts + pkt.eeg_counts,
                        "eeg_uV": [
                            round(v, 6) for v in (pending_board_pkt.eeg_uV + pkt.eeg_uV)
                        ],
                        "board_packet_id": pending_board_pkt.sample_id,
                        "daisy_packet_id": pkt.sample_id,
                        "stop_byte": f"0x{pkt.stop_byte:02X}",
                        "aux": pkt.aux,
                    }
                    pending_board_pkt = None
                    sample_count += 1
                yield record
    finally:
        ser.write(b"s")
        ser.flush()


def read_cyton_data(
        ser: serial.Serial,
        channels: int = 16,
        duration_s: Optional[float] = 5.0,
        max_samples: Optional[int] = None,
) -> List[Dict[str, object]]:
    return list(
        _iter_cyton_records(
            ser=ser,
            channels=channels,
            duration_s=duration_s,
            max_samples=max_samples,
        )
    )


def stream_packets(
        ser: serial.Serial,
        channels: int,
        duration_s: Optional[float],
        max_samples: Optional[int],
        output_jsonl: bool,
) -> int:
    sample_count = 0
    for record in _iter_cyton_records(
            ser=ser,
            channels=channels,
            duration_s=duration_s,
            max_samples=max_samples,
    ):
        sample_count += 1
        if output_jsonl:
            print(json.dumps(record, separators=(",", ":")))
        else:
            print(
                f"id={record['sample_id']:3d} "
                f"channels={len(record['eeg_uV'])} "
                f"eeg_uV={[round(v, 3) for v in record['eeg_uV']]} "
                f"aux={record['aux']}"
            )
    return sample_count


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read and parse OpenBCI Cyton packets from a USB serial port."
    )
    parser.add_argument("--port", default="/dev/ttyUSB0", help="Serial port path")
    parser.add_argument("--baud", type=int, default=115200, help="Serial baud rate")
    parser.add_argument(
        "--channels",
        type=int,
        choices=(8, 16),
        default=16,
        help="Set 8 for Cyton only, 16 for Cyton+Daisy",
    )
    parser.add_argument(
        "--probe-only",
        action="store_true",
        help="Only probe connection with 'v' command and exit",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="Streaming duration in seconds (omit with --max-samples for indefinite stream)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples to print (for 16ch this means combined samples)",
    )
    parser.add_argument("--timeout", type=float, default=0.5, help="Serial read timeout")
    parser.add_argument("--jsonl", action="store_true", help="Print packets as JSON lines")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    try:
        with serial.Serial(args.port, args.baud, timeout=args.timeout) as ser:
            probe_text = probe_connection(ser)
            if probe_text:
                print("Probe response:")
                print(probe_text)
            else:
                print("Probe response: <empty>")

            if args.probe_only:
                return 0

            mode_resp = set_channel_mode(ser, args.channels)
            if mode_resp:
                print(f"{args.channels}ch setup response:")
                print(mode_resp)

            duration_s = args.duration
            if duration_s is not None and duration_s <= 0:
                duration_s = None

            count = stream_packets(
                ser=ser,
                channels=args.channels,
                duration_s=duration_s,
                max_samples=args.max_samples,
                output_jsonl=args.jsonl,
            )
            print(f"Captured samples: {count}", file=sys.stderr)
            return 0
    except serial.SerialException as exc:
        print(f"Serial error on {args.port}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
