# Configuration for training the base Red Chamber model

# I/O
out_dir = 'out-red-chamber'
dataset = 'red_chamber'

# evaluation and logging
eval_interval = 250
eval_iters = 200
log_interval = 10
always_save_checkpoint = False

# wandb logging
wandb_log = False
wandb_project = 'red-chamber'
wandb_run_name = 'red-chamber-gpt'

# training
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256  # context window size
max_iters = 5000

# model architecture
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

# optimizer
learning_rate = 1e-3
lr_decay_iters = 5000
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 100

# system
# device = 'cpu'
# compile = False
