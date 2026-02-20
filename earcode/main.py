import numpy as np
import h5py
import scipy.io
import torch
import config as cfg
from train_valid_and_test import train_valid_model, test_model
from pathlib import Path
import csv




def from_mat_to_tensor(raw_data):
    #transpose, the dimention of mat and numpy is contrary
    Transpose = np.transpose(raw_data)
    Nparray = np.array(Transpose)
    return Nparray

def load_eeg_env(mat_path):
    try:
        with h5py.File(mat_path, 'r') as mat:
            return (
                from_mat_to_tensor(mat['EEG']),
                from_mat_to_tensor(mat['ENV'])
            )
    except (OSError, KeyError, ValueError):
        pass

    mat = scipy.io.loadmat(mat_path)
    if 'EEG' not in mat or 'ENV' not in mat:
        raise KeyError("Missing EEG/ENV variables in MAT file.")
    return np.array(mat['EEG']), np.array(mat['ENV'])

# all the number of sbjects in the experiment
# train one model for every subject

def run_pipeline(model_name=None):
    if model_name is not None:
        cfg.model_name = model_name

    eegname = str(Path(cfg.process_data_dir) / cfg.dataset_name)
    data, label = load_eeg_env(eegname)  # eeg data, labels (0/1) for attended direction

    torch.manual_seed(2024)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = bool(getattr(cfg, "deterministic", True))
        torch.backends.cudnn.benchmark = bool(getattr(cfg, "cudnn_benchmark", False))
        torch.cuda.manual_seed_all(2024)

    res = torch.zeros((cfg.sbnum, cfg.kfold_num))

    from sklearn.model_selection import KFold
    kfold = KFold(n_splits=cfg.kfold_num, shuffle=True, random_state=2024)

    for sb in range(cfg.sbnum):
        # get the data of specific subject
        eegdata = data[sb]
        eeglabel = label[sb]
        datasize = eegdata.shape
        eegdata = eegdata.reshape(4, 10, datasize[1], datasize[2])
        eeglabel = eeglabel.reshape(4, 10, datasize[1])
        # trans 10 and 4
        eegdata = np.transpose(eegdata, (1, 0, 2, 3))
        eeglabel = np.transpose(eeglabel, (1, 0, 2))

        for fold, (train_ids, test_ids) in enumerate(kfold.split(eegdata)):
            traindata = eegdata[train_ids]
            trainlabel = eeglabel[train_ids]
            testdata = eegdata[test_ids]
            testlabel = eeglabel[test_ids]
            traindata = traindata.reshape(
                32 * int(60 * cfg.sample_rate / cfg.decision_window),
                cfg.decision_window,
                20
            )
            trainlabel = trainlabel.reshape(
                32 * int(60 * cfg.sample_rate / cfg.decision_window),
                cfg.decision_window
            )
            testdata = testdata.reshape(
                8 * int(60 * cfg.sample_rate / cfg.decision_window),
                cfg.decision_window,
                20
            )
            testlabel = testlabel.reshape(
                8 * int(60 * cfg.sample_rate / cfg.decision_window),
                cfg.decision_window
            )

            train_valid_model(traindata, trainlabel, sb, fold)
            res[sb, fold] = test_model(testdata, testlabel, sb, fold)
        print("good job!")

    for sb in range(cfg.sbnum):
        print(sb)
        print(torch.mean(res[sb]))

    return res


if __name__ == "__main__":
    results = run_pipeline()
    # Save results next to this script so it's easy to locate regardless of CWD.
    out_csv = Path(__file__).resolve().parent / "result.csv"
    np.savetxt(str(out_csv), results.detach().cpu().numpy(), delimiter=",")

    # Also print and save a small human-readable summary.
    per_subject_mean = results.mean(dim=1).detach().cpu()
    overall_mean = per_subject_mean.mean().item()
    overall_std = per_subject_mean.std(unbiased=True).item() if per_subject_mean.numel() > 1 else 0.0

    print(f"\nSaved fold accuracies to: {out_csv}")
    print(f"Overall mean accuracy across {cfg.sbnum} subjects (%): {overall_mean:.4f}")
    print(f"Std across subjects (%): {overall_std:.4f}\n")

    out_summary = Path(__file__).resolve().parent / "result_summary.csv"
    with open(out_summary, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subject", "mean_accuracy_percent"])
        for sb, m in enumerate(per_subject_mean.tolist()):
            w.writerow([sb, float(m)])
        w.writerow([])
        w.writerow(["overall_mean_accuracy_percent", overall_mean])
        w.writerow(["std_across_subjects_percent", overall_std])
    print(f"Saved per-subject summary to: {out_summary}")


