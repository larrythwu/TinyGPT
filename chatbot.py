"""
Interactive chatbot using the trained model
"""
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out-wiki-zh-ft' # ignored if init_from is not 'resume'
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cpu' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
if init_from == 'resume':
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    # init from a given GPT-2 model
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)
if compile:
    model = torch.compile(model)

# look for the meta pickle in case it is available in the dataset folder
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']:
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

def generate_response(prompt):
    # Format the prompt with the new format
    formatted_prompt = f"<|prompt|>{prompt}<|response|>"
    
    # Encode the prompt
    input_ids = encode(formatted_prompt)
    x = torch.tensor(input_ids, dtype=torch.long, device=device)[None, ...]
    
    # Generate response
    with torch.no_grad():
        with ctx:
            y = model.generate(x, max_new_tokens=500, temperature=temperature, top_k=top_k)
            full_response = decode(y[0].tolist())
            
    # Extract only the response part
    try:
        # Find the start of response
        response_start = full_response.find('<|response|>') + len('<|response|>')
        # Find the end of response (only look for <|end|>)
        response_end = full_response.find('<|end|>', response_start)
        
        # Extract the response
        if response_end == -1:  # if <|end|> not found
            response = full_response[response_start:].strip()
        else:
            response = full_response[response_start:response_end].strip()
            
        if not response:
            response = "I couldn't generate a proper response. Please try again."
    except Exception:
        response = "I couldn't generate a proper response. Please try again."
    
    # Truncate full_response at the first <|end|> if it exists
    end_pos = full_response.find('<|end|>')
    if end_pos != -1:
        full_response = full_response[:end_pos + len('<|end|>')]
    
    return response, full_response

def main():
    print("Welcome to the Red Chamber Chatbot! (Press Ctrl+C to exit)")
    print("="*50)
    
    while True:
        try:
            user_input = input("\nYour message: ")
            if not user_input.strip():
                continue
                
            print("\nGenerating response...")
            response, full_response = generate_response(user_input)
            print("\nChatbot:", response)
            print("\nFull message with tags:")
            print(full_response)
            print("-"*50)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nAn error occurred: {str(e)}")
            continue

if __name__ == '__main__':
    main() 