import numpy as np
import torch

def load_token_data(path):
    """加载预处理的 token 数据（使用内存映射）"""
    return np.memmap(path, dtype=np.uint16, mode='r')

def get_batch(data, batch_size, seq_len, device):
    """
    从数据中随机采样一个 batch
    
    Args:
        data: np.memmap 或 np.ndarray, shape (total_tokens,)
        batch_size: 批次大小
        seq_len: 序列长度
        device: 设备 (cpu/cuda)
    
    Returns:
        x: (batch_size, seq_len) 输入 tokens
        y: (batch_size, seq_len) 目标 tokens (x 向右偏移一位)
    """
    # 1. 计算最大起始位置
    n = len(data) - seq_len - 1  # 需要 seq_len+1 个 token
    
    # 2. 随机选择 batch_size 个起始位置
    starts = torch.randint(0, n+1, (batch_size,))
    
    # 3. 直接取所有 chunk，不用循环
    # 从 data 中取出 batch_size 个长度为 seq_len+1 的序列
    chunks = [data[start:start + seq_len + 1] for start in starts]
    
    # 转成 tensor
    chunks = torch.tensor(np.stack(chunks).astype(np.int32), dtype=torch.long)
    # chunks: (batch_size, seq_len + 1)
    
    # 4. 切分输入和目标
    x = chunks[:, :seq_len]          # (batch_size, seq_len)
    y = chunks[:, 1:seq_len + 1]     # (batch_size, seq_len)
    
    # 5. 移到指定设备
    x = x.to(device)
    y = y.to(device)
    
    return x, y