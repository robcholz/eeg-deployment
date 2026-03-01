from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Iterator

__all__ = ["eeg_get_signal"]

_MODE = "offline"
_CURSOR = 0

_MODEL = None
_DEVICE = None
_NUM_CLASSES = 0
_OFFLINE_WINDOWS = None

_RT_SERIAL = None
_RT_RECORDS = None


def _import_earcode_modules():
    root_dir = Path(__file__).resolve().parent
    earcode_dir = root_dir / "earcode"
    if str(earcode_dir) not in sys.path:
        sys.path.insert(0, str(earcode_dir))

    import config as cfg  # type: ignore
    import main as ear_main  # type: ignore
    import model as ear_model  # type: ignore

    return cfg, ear_main, ear_model


def _build_model(cfg, ear_model):
    if cfg.model_name == "CNN_baseline":
        model = ear_model.CNN_baseline().to(cfg.device)
    elif cfg.model_name == "SANet":
        model = ear_model.EEG_SANet().to(cfg.device)
    elif cfg.model_name == "TANet":
        model = ear_model.EEG_TANet().to(cfg.device)
    elif cfg.model_name == "STANet":
        model = ear_model.EEG_STANet().to(cfg.device)
    elif cfg.model_name == "Transformer":
        model = ear_model.EEG_Transformer().to(cfg.device)
    elif cfg.model_name == "LinearTransformer":
        model = ear_model.EEG_LinearTransformer().to(cfg.device)
    else:
        raise ValueError(f"Unknown earcode model_name: {cfg.model_name}")
    model.eval()
    return model


def _ensure_model_loaded() -> None:
    global _MODEL, _DEVICE, _NUM_CLASSES
    if _MODEL is not None:
        return

    import torch  # type: ignore

    cfg, _, ear_model = _import_earcode_modules()
    subject_id = 0
    fold_id = 0
    earcode_dir = Path(__file__).resolve().parent / "earcode"
    ckpt_path = earcode_dir / "model_3D" / f"sb{subject_id}" / f"fold{fold_id}.ckpt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = _build_model(cfg=cfg, ear_model=ear_model)
    state_dict = torch.load(str(ckpt_path), map_location=cfg.device)
    model.load_state_dict(state_dict)
    model.eval()

    _MODEL = model
    _DEVICE = cfg.device
    _NUM_CLASSES = int(getattr(cfg, "categorie_num", 4))


def _load_offline_windows():
    import numpy as np  # type: ignore

    cfg, ear_main, _ = _import_earcode_modules()
    subject_id = 0
    eeg_path = str(Path(cfg.process_data_dir) / cfg.dataset_name)
    data, _ = ear_main.load_eeg_env(eeg_path)

    eegdata = data[subject_id]
    datasize = eegdata.shape
    eegdata = eegdata.reshape(4, 10, datasize[1], datasize[2])
    eegdata = np.transpose(eegdata, (1, 0, 2, 3))
    windows = eegdata.reshape(
        40 * int(60 * cfg.sample_rate / cfg.decision_window),
        cfg.decision_window,
        20,
    )
    if windows.shape[0] == 0:
        raise RuntimeError("No EEG windows available after reshape")
    return windows


def _ensure_offline_windows() -> None:
    global _OFFLINE_WINDOWS
    if _OFFLINE_WINDOWS is None:
        _OFFLINE_WINDOWS = _load_offline_windows()


def _adapt_16ch_125hz_to_20ch_128hz(samples_16ch_125: "object") -> "object":
    import numpy as np  # type: ignore

    x = np.asarray(samples_16ch_125, dtype=np.float32)
    if x.shape != (125, 16):
        raise ValueError(f"Expected shape (125, 16), got {x.shape}")

    old_t = np.arange(125, dtype=np.float32)
    new_t = np.arange(128, dtype=np.float32) * (124.0 / 127.0)
    resampled = np.empty((128, 16), dtype=np.float32)
    for ch in range(16):
        resampled[:, ch] = np.interp(new_t, old_t, x[:, ch]).astype(np.float32)

    out = np.zeros((128, 20), dtype=np.float32)
    out[:, :16] = resampled
    return out


def _ensure_realtime_stream() -> None:
    global _RT_SERIAL, _RT_RECORDS
    if _RT_SERIAL is not None and _RT_RECORDS is not None:
        return

    import serial  # type: ignore
    import cyton_usb_reader as cyton

    _RT_SERIAL = serial.Serial("/dev/ttyUSB0", 115200, timeout=0.5)
    cyton.probe_connection(_RT_SERIAL)
    cyton.set_channel_mode(_RT_SERIAL, 16)
    _RT_RECORDS = cyton._iter_cyton_records(  # noqa: SLF001
        ser=_RT_SERIAL,
        channels=16,
        duration_s=None,
        max_samples=None,
    )


def _next_realtime_window_20x128() -> "object":
    import numpy as np  # type: ignore

    _ensure_realtime_stream()
    assert _RT_RECORDS is not None

    rows = []
    while len(rows) < 125:
        rec = next(_RT_RECORDS)
        eeg_uV = rec.get("eeg_uV")
        if not isinstance(eeg_uV, list):
            continue
        if len(eeg_uV) != 16:
            continue
        rows.append([float(v) for v in eeg_uV])

    block_125x16 = np.asarray(rows, dtype=np.float32)
    return _adapt_16ch_125hz_to_20ch_128hz(block_125x16)


def _infer_probs(window_128x20: "object") -> list[float]:
    import numpy as np  # type: ignore
    import torch  # type: ignore

    _ensure_model_loaded()
    assert _MODEL is not None
    assert _DEVICE is not None

    x = np.asarray(window_128x20, dtype=np.float32)
    if x.shape != (128, 20):
        raise ValueError(f"Expected shape (128, 20), got {x.shape}")

    tensor = torch.tensor(x, dtype=torch.float32, device=_DEVICE).unsqueeze(0)
    with torch.no_grad():
        logits = _MODEL(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().tolist()
    return [float(v) for v in probs]


def eeg_get_signal() -> list[float]:
    """
    Public API: return one n-class EEG vector.
    Default mode reads from earcode offline windows.
    Real-time mode reads Cyton stream and adapts 16ch@125Hz -> 20ch@128Hz.
    """
    global _CURSOR
    if _MODE == "real-time":
        window = _next_realtime_window_20x128()
        return _infer_probs(window)

    _ensure_offline_windows()
    assert _OFFLINE_WINDOWS is not None
    window = _OFFLINE_WINDOWS[_CURSOR % _OFFLINE_WINDOWS.shape[0]]
    _CURSOR += 1
    return _infer_probs(window)


def _close_realtime_stream() -> None:
    global _RT_RECORDS, _RT_SERIAL
    if _RT_RECORDS is not None:
        try:
            _RT_RECORDS.close()
        except Exception:
            pass
        _RT_RECORDS = None
    if _RT_SERIAL is not None:
        try:
            _RT_SERIAL.write(b"s")
            _RT_SERIAL.flush()
        except Exception:
            pass
        try:
            _RT_SERIAL.close()
        except Exception:
            pass
        _RT_SERIAL = None


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="EEG signal API demo")
    parser.add_argument(
        "--real-time",
        action="store_true",
        help="Use cyton_usb_reader stream with adapter (16ch@125Hz -> 20ch@128Hz).",
    )
    return parser


def main() -> int:
    global _MODE
    args = _build_arg_parser().parse_args()

    if args.real_time:
        _MODE = "real-time"
        print("real-time mode started (Ctrl+C to stop)")
        idx = 0
        try:
            while True:
                print(f"{idx}: {eeg_get_signal()}")
                idx += 1
        except KeyboardInterrupt:
            print("\nstopped")
        finally:
            _close_realtime_stream()
        return 0

    _MODE = "offline"
    print("offline mode demo:")
    for i in range(8):
        print(f"{i}: {eeg_get_signal()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
