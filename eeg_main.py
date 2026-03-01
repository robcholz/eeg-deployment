from __future__ import annotations

from pathlib import Path
import sys

__all__ = ["eeg_get_signal"]

_CURSOR = 0
_WINDOWS = None
_MODEL = None
_DEVICE = None
_NUM_CLASSES = 0


def _build_model(cfg, ear_model):
    import torch  # type: ignore

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


def _prepare_windows(data, sb: int, cfg):
    import numpy as np  # type: ignore

    eegdata = data[sb]
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


def _ensure_loaded() -> None:
    global _WINDOWS, _MODEL, _DEVICE, _NUM_CLASSES
    if _MODEL is not None and _WINDOWS is not None:
        return

    import torch  # type: ignore

    root_dir = Path(__file__).resolve().parent
    earcode_dir = root_dir / "earcode"
    if str(earcode_dir) not in sys.path:
        sys.path.insert(0, str(earcode_dir))

    import config as cfg  # type: ignore
    import main as ear_main  # type: ignore
    import model as ear_model  # type: ignore

    subject_id = 0
    fold_id = 0

    eeg_path = str(Path(cfg.process_data_dir) / cfg.dataset_name)
    data, _ = ear_main.load_eeg_env(eeg_path)
    windows = _prepare_windows(data=data, sb=subject_id, cfg=cfg)

    model = _build_model(cfg=cfg, ear_model=ear_model)
    ckpt_path = earcode_dir / "model_3D" / f"sb{subject_id}" / f"fold{fold_id}.ckpt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    state_dict = torch.load(str(ckpt_path), map_location=cfg.device)
    model.load_state_dict(state_dict)
    model.eval()

    _WINDOWS = windows
    _MODEL = model
    _DEVICE = cfg.device
    _NUM_CLASSES = int(getattr(cfg, "categorie_num", 4))


def eeg_get_signal() -> list[float]:
    """
    Return one EEG decoder sample as an n-class probability list.
    This function strictly runs earcode model inference (no fallback path).
    """
    global _CURSOR
    _ensure_loaded()
    assert _WINDOWS is not None
    assert _MODEL is not None
    assert _DEVICE is not None

    import numpy as np  # type: ignore
    import torch  # type: ignore

    window = _WINDOWS[_CURSOR % _WINDOWS.shape[0]]
    _CURSOR += 1

    x = torch.tensor(np.asarray(window), dtype=torch.float32, device=_DEVICE).unsqueeze(0)
    with torch.no_grad():
        logits = _MODEL(x)
        probs = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().tolist()
    return [float(v) for v in probs]


if __name__ == "__main__":
    print("eeg_get_signal demo:")
    for i in range(8):
        print(f"{i}: {eeg_get_signal()}")
