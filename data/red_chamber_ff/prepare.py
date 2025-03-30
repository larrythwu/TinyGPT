import re
import os
import tiktoken
import numpy as np

# Load the provided text (assuming it's saved as 'hongloumeng_excerpt.txt')
with open(os.path.join(os.path.dirname(__file__), 'DORC.txt'), 'r', encoding='utf-8') as file:
    text = file.read()

# Function to extract conversations intelligently
def extract_conversations(text):
    # Pattern for explicit quoted dialogue
    quoted_pattern = r'[“"](.*?)[”"]\s*(?:[^“”"]*?(?:说|道|问|笑道|回|叹|叫|念|哭|喊|唱|曰|云|言|问曰|答曰|答道|回道|说道|叹道|叫道|念道|哭道)[:：]?)?'
    
    # Pattern for implicit dialogue (speech verbs without quotes)
    implicit_pattern = r'(?:说|道|问|笑道|回|叹|叫|念|哭|喊|唱|曰|云|言|问曰|答曰|答道|回道|说道|叹道|叫道|念道|哭道)[:：]?\s*([^“”"\n。！？]+?)(?=\s*(?:[。！？]|说|道|问|笑道|回|叹|叫|念|哭|喊|唱|曰|云|言|问曰|答曰|答道|回道|说道|叹道|叫道|念道|哭道|\n))'
    
    # Find all dialogues
    quoted_dialogues = re.findall(quoted_pattern, text, re.DOTALL)
    implicit_dialogues = re.findall(implicit_pattern, text)
    
    # Clean and combine dialogues
    dialogues = []
    for d in quoted_dialogues:
        cleaned = d.strip()
        if cleaned and len(cleaned) > 1:  # Ignore very short fragments
            dialogues.append(cleaned)
    
    for d in implicit_dialogues:
        cleaned = d.strip()
        if cleaned and len(cleaned) > 1 and not cleaned.startswith('“') and not cleaned.startswith('"'):
            dialogues.append(cleaned)
    
    # Pair consecutive dialogues into prompt-response
    conversations = []
    i = 0
    while i < len(dialogues) - 1:
        prompt = dialogues[i]
        response = dialogues[i + 1]
        
        # Check if they seem like a conversational exchange (basic heuristic)
        # Avoid pairing if response is too short or seems like narration
        if len(response) > 2 and not re.match(r'^[于是|且说|忽|那|只见]', response):
            conversations.append((prompt, response))
            i += 2  # Skip the response to avoid overlap
        else:
            i += 1
    
    return conversations

# Extract conversations
conversations = extract_conversations(text)

# Format into prompt-response structure
formatted_conversations = []
for prompt, response in conversations:
    formatted = f"<prompt>{prompt}</prompt>\n<response>{response}</response>\n"
    formatted_conversations.append(formatted)

# Write to output file
output_file = os.path.join(os.path.dirname(__file__), 'chat_data.txt')
with open(output_file, 'w', encoding='utf-8') as f:
    f.write("\n".join(formatted_conversations))

print(f"Extracted {len(formatted_conversations)} conversations and saved to {output_file}")

data = "\n".join(formatted_conversations)
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# encode with tiktoken gpt2 bpe
enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)
# decode and print first 10 tokens
print("First 10 tokens decoded:", enc.decode(train_ids[:10]))
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))