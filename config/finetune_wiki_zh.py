# Configuration for fine-tuning the Wiki Zh model

import time

# I/O
out_dir = 'out-wiki-zh-ft'
dataset = 'silk_road_ft'

wandb_log = True
wandb_project = 'owt'
wandb_run_name='wiki_zh_gpt2-124M-ft'
init_from = 'resume'

# evaluation and logging
eval_interval = 50
eval_iters = 200
log_interval = 10
always_save_checkpoint = False

# training
batch_size = 12
block_size = 1024
gradient_accumulation_steps = 1

max_iters = 60000

# optimizer
learning_rate = 3e-5
decay_lr = False