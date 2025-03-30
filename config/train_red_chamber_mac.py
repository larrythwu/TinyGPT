# Configuration for training the Red Chamber model on MacBook
# Reduced model size and batch size for CPU training

# I/O
out_dir = 'out-red-chamber'
dataset = 'red_chamber'

# evaluation and logging
eval_interval = 250
eval_iters = 20
log_interval = 1
always_save_checkpoint = False

# wandb logging
wandb_log = False
wandb_project = 'red-chamber-gpt'
wandb_run_name = 'red-chamber-base'

# training
gradient_accumulation_steps = 1
batch_size = 12
block_size = 64  # context window size
max_iters = 2000

# model architecture
n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.0

# optimizer
learning_rate = 1e-3
lr_decay_iters = 2000
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 100

# system
device = 'cpu'
compile = False
