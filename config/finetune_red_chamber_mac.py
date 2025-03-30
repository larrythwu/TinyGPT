# Configuration for fine-tuning the Red Chamber model on MacBook
# Reduced context size for CPU training

import time

# I/O
out_dir = 'out-red-chamber'
dataset = 'red_chamber_ft'
init_from = 'resume'

# evaluation and logging
eval_interval = 250
eval_iters = 20
log_interval = 1
always_save_checkpoint = False

# wandb logging
wandb_log = False
wandb_project = 'red-chamber-gpt'
wandb_run_name = 'red-chamber-ft'

# training
gradient_accumulation_steps = 1
batch_size = 32
block_size = 64  # context window size
max_iters = 6000 # this includes the 2000 in the original config

# optimizer
learning_rate = 3e-5
decay_lr = False

# system
device = 'cpu'
compile = False