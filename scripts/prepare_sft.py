import json
import os
import hashlib
import argparse
import numpy as np
import torch
from tokenizers import Tokenizer


def render_chat(conversations):
    """
    input:  [{"role": "user", "content": "你好"},
             {"role": "assistant", "content": "你好！"}]
    output: "### Human: 你好\n### Assistant: 你好！"
    """
    lines = []
    for msg in conversations:
        role = msg['role']
        content = msg['content']
        if role == 'user':
            lines.append(f"### Human: {content}")
        elif role == 'assistant':
            lines.append(f"### Assistant: {content}")
        # reasoning_content 直接忽略
    return '\n'.join(lines)


def tokenize_with_mask(tokenizer, rendered_text, seq_len=512):
    """
    rendered_text: "### Human: xxx\n### Assistant: yyy"
    tokenizer: 预训练时训练好的 BPE

    返回:
        input_ids: 整段文本的 token ID
        labels:    Human 部分 = -100, Assistant 部分 = 真实 ID
    """
    full_input_ids = []
    full_labels = []
    
    for line in rendered_text.split('\n'):
        if not line.strip():
            continue
        ids = tokenizer.encode(line).ids
        
        full_input_ids.extend(ids)
        
        if line.startswith('### Human:'):
            full_labels.extend([-100] * len(ids))
        else:  # Assistant 或其他
            full_labels.extend(ids)
    
    # 截断/padding 到 seq_len
    if len(full_input_ids) > seq_len:
        # 截断
        full_input_ids = full_input_ids[:seq_len]
        full_labels = full_labels[:seq_len]
    else:
        # Padding 到 seq_len
        pad_len = seq_len - len(full_input_ids)
        full_input_ids.extend([0] * pad_len)          # 用 0 作为 pad token ID
        full_labels.extend([-100] * pad_len)          # pad 部分不算 loss
    
    # 转成 tensor
    input_ids = torch.tensor(full_input_ids, dtype=torch.long)
    labels = torch.tensor(full_labels, dtype=torch.long)
    
    return input_ids, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/minimind_sft/sft_t2t_mini.jsonl')
    parser.add_argument('--tokenizer-path', default='data/pretrain_clean/tokenizer.json')
    parser.add_argument('--output-dir', default='data/minimind_sft/sft_clean')
    parser.add_argument('--seq-len', type=int, default=512)
    parser.add_argument('--train-ratio', type=float, default=0.99)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer = Tokenizer.from_file(args.tokenizer_path)

    # 1. 先数总行数
    print("Counting total samples...")
    n_total = 0
    with open(args.input, 'r', encoding='utf-8') as f:
        for line in f:
            n_total += 1
    print(f"Total samples: {n_total}")
    
    # 2. 预分配memmap，逐条写入磁盘
    data = np.memmap(
        os.path.join(args.output_dir, 'all_sft.bin'),
        dtype=np.int32, mode='w+',
        shape=(n_total, 2, args.seq_len)
    )

    hashes = []
    with open(args.input, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i % 100000 == 0:
                print(f"Processing {i}/{n_total}...")

            obj = json.loads(line.strip())
            rendered = render_chat(obj['conversations'])
            input_ids, labels = tokenize_with_mask(tokenizer, rendered, args.seq_len)

            data[i, 0] = input_ids.numpy()
            data[i, 1] = labels.numpy()
            hashes.append(hashlib.sha1(rendered.encode('utf-8')).hexdigest())
            
    # 3. SHA1 划分 train/valid
    print("Splitting train/valid...")
    train_indices = []
    valid_indices = []

    for i, h in enumerate(hashes):
        hash_int = int(h[:8], 16)
        if hash_int % 100 < 99:
            train_indices.append(i)
        else:
            valid_indices.append(i)

    # 4. 从 memmap 提取 train/valid 并保存
    print(f"Saving train set ({len(train_indices)} samples)...")
    train_data = data[train_indices]
    train_data.tofile(os.path.join(args.output_dir, 'train_sft.bin'))

    print(f"Saving valid set ({len(valid_indices)} samples)...")
    valid_data = data[valid_indices]
    valid_data.tofile(os.path.join(args.output_dir, 'valid_sft.bin'))
    
     # 5. 保存元信息+清理中间文件
    with open(os.path.join(args.output_dir, 'sft_meta.json'), 'w') as f:
        json.dump({
            'n_train': len(train_indices),
            'n_valid': len(valid_indices),
            'seq_len': args.seq_len,
        }, f)

    os.remove(os.path.join(args.output_dir, 'all_sft.bin'))

    print(f"Done. Train: {len(train_indices)} samples, Valid: {len(valid_indices)} samples")
    
if __name__ == '__main__':
    main()