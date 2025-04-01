from datasets import load_dataset
from huggingface_hub import login
import os
import tiktoken
import numpy as np

login("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")

# Step 1: Load the dataset
dataset = load_dataset("silk-road/Luotuo-QA-A-CoQA-Chinese")

# Step 2: Extract and format all Q&As
output_lines = []
for example in dataset["train"]:  # Adjust split name if needed (e.g., "train", "validation")
    questions = example["questions_zh"]  # List of Chinese questions
    answers = example["answers_zh"]      # List of Chinese answers
    for q, a in zip(questions, answers):
        line = f"<|prompt|>{q}<|response|>{a}<|end|>"
        output_lines.append(line)

data = "\n".join(output_lines)
print(data[:200])
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# encode with tiktoken gpt2 bpe
enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)
# decode and print first 10 tokens
print("First 10 tokens decoded:\n", enc.decode(train_ids[:200]))
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))