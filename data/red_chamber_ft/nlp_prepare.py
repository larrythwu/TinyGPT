import re
import jieba
import os
import tiktoken
import numpy as np

# 扩展的人物字典
characters = {
    "宝玉": "贾宝玉", "黛玉": "林黛玉", "宝钗": "薛宝钗", "凤姐": "王熙凤",
    "贾母": "贾母", "史氏": "史太君", "士隐": "甄士隐", "雨村": "贾雨村",
    "英莲": "甄英莲", "冯氏": "冯氏", "探春": "贾探春", "迎春": "贾迎春",
    "惜春": "贾惜春", "元春": "贾元春", "湘云": "史湘云", "妙玉": "妙玉",
    "晴雯": "晴雯", "袭人": "袭人", "平儿": "平儿", "紫鹃": "紫鹃",
    "刘姥姥": "刘姥姥", "巧姐": "贾巧姐", "鸳鸯": "鸳鸯", "薛蟠": "薛蟠",
    "贾琏": "贾琏", "王夫人": "王夫人", "尤氏": "尤氏", "李纨": "李纨",
}

# 输入和输出文件
input_file = os.path.join(os.path.dirname(__file__), "DORC.txt")
output_file = os.path.join(os.path.dirname(__file__), "nlp_chat_data.txt")

# 读取文本文件
def load_text(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

# 使用 jieba 分词识别人物
def identify_speaker(context):
    words = jieba.cut(context)
    for word in words:
        for char_short, char_full in characters.items():
            if char_short in word or char_full in word:
                return char_full
    return "未知人物"

# 判断是否为“问”
def is_question(dialogue):
    return "?" in dialogue or "吗" in dialogue or "何" in dialogue or "怎" in dialogue

# 提取对话并确保成对
def extract_dialogues(text):
    dialogue_pattern = r"([^\n。]+?)道：“([^”]+)”"
    dialogues = re.findall(dialogue_pattern, text)
    
    qa_pairs = []
    i = 0
    while i < len(dialogues) - 1:  # 确保有下一句可配对
        current_context, current_dialogue = dialogues[i]
        next_context, next_dialogue = dialogues[i + 1]
        
        current_speaker = identify_speaker(current_context)
        next_speaker = identify_speaker(next_context)
        
        # 当前是“问”，且下一句是不同人物的“答”
        if is_question(current_dialogue) and not is_question(next_dialogue) and current_speaker != next_speaker:
            qa_pairs.append("<prompt>" + current_dialogue.replace('\n','') + "</prompt>")
            qa_pairs.append("<response>" + next_dialogue.replace('\n','') + "</response>")
            i += 2  # 跳过已配对的两句
        else:
            i += 1  # 跳过独白或不匹配的对话
    
    return qa_pairs

# 保存结果到文件
def save_to_file(qa_pairs, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for i in range(0, len(qa_pairs), 2):  # 每两行为一对
            f.write(f"{qa_pairs[i]}\n")
            f.write(f"{qa_pairs[i + 1]}\n")

# 主函数
def main():
    text = load_text(input_file)
    qa_pairs = extract_dialogues(text)
    
    print("提取的前5个问答对：")
    for pair in qa_pairs[:5]:
        print(pair)
    
    save_to_file(qa_pairs, output_file)
    print(f"结果已保存到 {output_file}")
    
    data = "\n".join(qa_pairs)
    print(data[:100])
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

if __name__ == "__main__":
    jieba.initialize()
    main()