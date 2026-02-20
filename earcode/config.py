import torch
from pathlib import Path

# models
model_names = ['CNN_baseline', 'SANet', 'TANet', 'STANet', 'Transformer', 'LinearTransformer']
model_name = model_names[0]  # you could change the code to other models by only changing the number
_EARCODE_DIR = Path(__file__).resolve().parent
# Data file `EAR_4_direction_1D.mat` is expected to be placed in `analysis/python/`
process_data_dir = str(_EARCODE_DIR.parent)
dataset_name = 'EAR_4_direction_1D.mat'

device_ids = 0
device = torch.device(f"cuda:{device_ids}" if torch.cuda.is_available() else "cpu")
epoch_num = 100
batch_size = 128
sample_rate = 128
categorie_num = 4
sbnum = 16
kfold_num = 5

lr=1e-3
weight_decay=0.01

# the length of decision window
decision_window = 128 #1s

# performance knobs (CUDA only)
use_amp = True          # enable autocast + GradScaler during training
allow_tf32 = True       # allow TF32 matmul on Ampere+ GPUs (often faster, tiny precision change)
deterministic = True    # reproducible results (may reduce speed)
cudnn_benchmark = False # set True for speed when deterministic=False
use_compile = False     # torch.compile (PyTorch 2.x) for speed
compile_mode = "reduce-overhead"
compile_fullgraph = False

# dataloader knobs
num_workers = 2
pin_memory = True

# transformer hyperparameters
transformer_d_model = 16
transformer_nhead = 1
transformer_num_layers = 1
transformer_ff_dim = 32
transformer_dropout = 0.1



