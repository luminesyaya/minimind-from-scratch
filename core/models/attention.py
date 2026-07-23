import torch
import math

def precompute_rope_freqs(d_head, max_seq_len, theta=10000.0):
    """
    预计算 cos 和 sin 表
    返回: cos, sin 各 (max_seq_len, d_head // 2)
    """
    freqs = 1.0 / (theta ** (torch.arange(0, d_head, 2).float() / d_head))
    # freqs shape: (d_head // 2,)
    t = torch.arange(max_seq_len)    
    
    angles = torch.outer(t, freqs)  # (max_seq_len, d_head // 2)
    
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    
    return cos, sin

def apply_rotary_emb(x, cos, sin, offset=0):
    """
    x: (batch, seq_len, num_heads, d_head)
    cos, sin: (max_seq_len, d_head//2)
    offset: KV Cache 时跳过已有位置
    """
    x_even = x[..., 0::2]
    x_old = x[..., 1::2]
    
    seq_len = x.shape[1]
    cos = cos[offset: offset + seq_len].unsqueeze(0).unsqueeze(2) # (1, seq_len, 1, d_head//2)
    sin = sin[offset: offset + seq_len].unsqueeze(0).unsqueeze(2) # (1, seq_len, 1, d_head//2)
    
    x_rotated_even = x_even * cos - x_old * sin
    x_rotated_odd = x_old * cos + x_even * sin
    
    x_rotated = torch.stack([x_rotated_even, x_rotated_odd], dim=-1).flatten(start_dim=-2)
    return x_rotated

def repeat_kv(kv, n_rep: int):
    """
    kv: (batch, seq_len, num_kv_heads, d_head)
    n_rep: 每个 KV 头要服务几个 Q 头
    """
    if n_rep == 1:
        return kv
    return kv.repeat_interleave(n_rep, dim=2)

def gqa_attention(q, k, v, cos, sin, past_kv=None, past_len=0, mask=None):
    """
    q: (batch, seq_len, num_q_heads, d_head)
    k: (batch, seq_len, num_kv_heads, d_head)
    v: (batch, seq_len, num_kv_heads, d_head)
    past_kv: (past_seq_len, num_kv_heads, d_head) 或 None
    past_len: 已缓存的长度
    mask: (batch, seq_len, seq_len) 或 None
    """
    
    n_rep = q.shape[2] // k.shape[2]
    
    q = apply_rotary_emb(q, cos, sin, offset=past_len)
    k = apply_rotary_emb(k, cos, sin, offset=past_len)
    
    if past_kv is not None:
        past_k, past_v = past_kv
        k = torch.cat([past_k, k], dim=1)
        v = torch.cat([past_v, v], dim=1)
    
    new_kv = (k, v)
    
    k = repeat_kv(k, n_rep)
    v = repeat_kv(v, n_rep)
    
    # (batch, seq_len, num_heads, d_head) → (batch, num_heads, seq_len, d_head)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    
    d_head = q.shape[-1]
    scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(d_head)
    
    if mask is not None:
        scores = scores + mask
        
    attn_weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, v)  # (batch, num_heads, seq_len, d_head)
    
    output = output.transpose(1, 2)  # (batch, seq_len, num_heads, d_head)
    return output, new_kv
    