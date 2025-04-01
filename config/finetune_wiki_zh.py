# Configuration for fine-tuning the Red Chamber model

import time

# I/O
out_dir = 'out-wiki-zh-ft'
dataset = 'red_chamber_ft'

wandb_log = True
wandb_project = 'owt'
wandb_run_name='wiki_zh_gpt2-124M-ft'
init_from = 'resume'

# evaluation and logging
eval_interval = 5
eval_iters = 1
log_interval = 1
always_save_checkpoint = False

# training
batch_size = 12
block_size = 1024
gradient_accumulation_steps = 1

max_iters = 5000

# optimizer
learning_rate = 3e-5
decay_lr = False