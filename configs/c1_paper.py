"""Default config for C1 experiments.

This config is a paper-oriented starting point.
Adjust `batch_size` and `gradient_accumulation_steps` to your hardware.
"""

# --- dataset ---
dataset = "openwebtext"
data_dir = "data/openwebtext"

# --- model ---
# GPT-2 small-ish
n_layer = 12
n_head = 12
n_embd = 768
block_size = 1024
bias = False
dropout = 0.0

# --- batch ---
batch_size = 8
gradient_accumulation_steps = 6  # tokens/step = batch_size*block_size*grad_accum

# --- training ---
max_iters = 2000
eval_interval = 200
eval_iters = 50
log_interval = 10

# LR schedule
learning_rate = 3e-4
min_lr = 3e-5
warmup_iters = 200
lr_decay_iters = 2000
lr_schedule = "cosine"  # "cosine" | "linear"

# Which optimizer to use by default.
# You can always override at launch time via:  python train.py ... --optimizer <name>
optimizer = "sophiag"  # "adamw" | "lion" | "sophiag" | "lamb" | "muon" | "sgd" | "selector"


# stability
grad_clip = 1.0
weight_decay = 0.1  # decoupled, applied externally

# optimizer-specific defaults (you can override via CLI)
adamw_beta1 = 0.9
adamw_beta2 = 0.95
adamw_eps = 1e-8

sgd_momentum = 0.9
sgd_nesterov = False

lion_beta1 = 0.9
lion_beta2 = 0.99

lamb_beta1 = 0.9
lamb_beta2 = 0.999
lamb_eps = 1e-6

# SophiaG
sophia_beta1 = 0.965
sophia_beta2 = 0.99
sophia_rho = 0.04
sophia_eps = 1e-12
sophia_update_hess_interval = 10

# Muon (rough defaults; check official repo docs if you want to tune)
muon_momentum = 0.95
muon_beta1 = 0.9
muon_beta2 = 0.95
muon_eps = 1e-8

# --- diagnostics ---
diag = True
diag_every = 100
diag_probes = 4
# heavy-tail samples per diag tick (kept moderate; increase if you want smoother CCDF)
tail_samples = 80000

# --- selector ---
selector_candidates = ["adamw", "lion", "lamb", "muon", "sophiag", "sgd"]
sel_every = 100
sel_min_ep = 100
sel_patience = 100
sel_dc_ema_rho = 0.1
sel_dc_margin = 0.05
sel_dc_min = 0.0
sel_dc_probes = 4

# --- heavy-tail diagnostics: gradient + gradient-noise tails ---
# Uses K independent mini-batches to estimate mean grad and noise. See tools/plot_tail_metrics.py
tail_diag = True
tail_every = 100
tail_k_batches = 32
# number of sampled coordinates per group per diagnostic tick (increase for smoother tails)
tail_samples_per_group = 200000
tail_groups = ["all", "decay", "no_decay", "norm", "bias", "embed"]
tail_save_every = 1  # save arrays every tick; increase to reduce disk
tail_seed = 1337
tail_mean_estimator = "mean"  # "mean" | "mom"
tail_mom_chunks = 8
tail_fp32 = True
tail_log_delta = True
tail_log_delta_signed = False  # set True for alpha-stable "stability under sums" test
tail_delta_signed_samples = 50000
tail_grouping_mode = "basic"  # "basic" or "layer_bucket"
