import numpy as np
import torch


def load_sft_data(path, shape):
    """
    path: train_sft.bin 路径
    shape: (n_samples, 2, seq_len) — 从 meta.json 读取
    """
    data = np.memmap(path, dtype=np.int32, mode='r')
    return data.reshape(shape)


def get_sft_batch(data, batch_size, device):
    """
    data: (n_samples, 2, seq_len)
    随机选 batch_size 条完整对话
    返回: (x, y)
        x: (batch, seq_len) input_ids
        y: (batch, seq_len) labels (Human 部分 = -100)
    """
    n = data.shape[0]
    indices = torch.randint(0, n, (batch_size,))
    
    x = torch.tensor(data[indices, 0].astype(np.int64), dtype=torch.long).to(device)
    y = torch.tensor(data[indices, 1].astype(np.int64), dtype=torch.long).to(device)
    return x, y