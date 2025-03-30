import time

out_dir = 'out-red-chamber'
eval_interval = 250 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 1 # don't print too too often
max_iters = 1000
wandb_log = False # feel free to turn on
wandb_project = 'red_chamber_ff'
wandb_run_name = 'mini-gpt'

dataset = 'red_chamber_ff'
init_from = 'resume'

# only save checkpoints if the validation loss improves
always_save_checkpoint = False

dataset = 'red_chamber_ff'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 64  # reduced to match base model's block size

# finetune at constant LR
learning_rate = 3e-5
decay_lr = False

# on macbook also add
device = 'cpu'  # run on cpu only
compile = False # do not torch compile the model