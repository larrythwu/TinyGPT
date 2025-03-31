# Configuration for fine-tuning the Red Chamber model

import time

# I/O
out_dir = 'out-red-chamber'
dataset = 'red_chamber_ft'
init_from = 'resume'

# evaluation and logging
eval_interval = 250
eval_iters = 200
log_interval = 10
always_save_checkpoint = False

# wandb logging
wandb_log = False
wandb_project = 'red-chamber-gpt'
wandb_run_name = 'red-chamber-ft'

# training
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256  # context window size
max_iters = 5000

# optimizer
learning_rate = 3e-5
decay_lr = False

# system
# device = 'cpu'
# compile = False