import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .norm import RMSNorm
from .ffn import SwiGLUFFN
from .attention import gqa_attention, precompute_rope_freqs

class DecoderBlock(nn.Module):
    def __init__(self, dim, num_q_heads, num_kv_heads, ffn_multiplier=4):
        super().__init__()
        self.dim = dim
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.dim_head = dim // num_q_heads
        
        self.q_proj = nn.Linear(dim, num_q_heads * self.dim_head, bias=False)
        self.k_proj = nn.Linear(dim, num_kv_heads * self.dim_head, bias=False)
        self.v_proj = nn.Linear(dim, num_kv_heads * self.dim_head, bias=False)
        self.out_proj = nn.Linear(num_q_heads * self.dim_head, dim, bias=False)
        
        hidden_dim = int(dim * ffn_multiplier)
        self.ffn = SwiGLUFFN(dim, hidden_dim)
        
        self.attn_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)
        
    def forward(self, x, cos, sin, past_kv=None, mask=None):
        """
        Args:
            x: (batch, seq_len, dim) 当前输入
            cos: (max_seq_len, d_head//2) 预计算的 cos 表
            sin: (max_seq_len, d_head//2) 预计算的 sin 表
            past_kv: 元组 (past_k, past_v) 或 None
                     past_k: (batch, past_seq_len, num_kv_heads, d_head)
                     past_v: (batch, past_seq_len, num_kv_heads, d_head)
            mask: (batch, seq_len, seq_len) 或 None, 因果掩码
        
        Returns:
            x: (batch, seq_len, dim) 输出
            new_kv: 元组 (new_k, new_v)
                    new_k: (batch, past_seq_len + seq_len, num_kv_heads, d_head)
                    new_v: (batch, past_seq_len + seq_len, num_kv_heads, d_head)
        """
        batch_size, seq_len, _ = x.shape
        
        residual = x
        x_normed = self.attn_norm(x)
        
        q = self.q_proj(x_normed)
        k = self.k_proj(x_normed)
        v = self.v_proj(x_normed)
        
        q = q.view(batch_size, seq_len, self.num_q_heads, self.dim_head)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.dim_head)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.dim_head)
        
        if mask is None and seq_len > 1 and past_kv is None:
            mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device) * float('-inf'),
                diagonal=1
            ).unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
            
        
        past_len = past_kv[0].shape[1] if past_kv is not None else 0
        attn_out, new_kv = gqa_attention(q, k, v, cos, sin, past_kv=past_kv, past_len=past_len, mask=mask)
        
        attn_out = attn_out.contiguous().view(batch_size, seq_len, -1)
        attn_out = self.out_proj(attn_out)
        x= residual + attn_out
        
        residual = x
        x_normed = self.ffn_norm(x)
        ffn_out = self.ffn(x_normed)
        x = residual + ffn_out
        
        return x, new_kv
        
class MiniLLM(nn.Module):
    def __init__(self, vocab_size, dim, num_q_heads, num_kv_heads,
                 num_layers, max_seq_len=2048):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.d_head = dim // num_q_heads   
        
        self.embed = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            DecoderBlock(dim, num_q_heads, num_kv_heads)
            for _ in range(num_layers)
        ])    
        self.final_norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        
        cos, sin = precompute_rope_freqs(self.d_head, max_seq_len)
        self.register_buffer('cos', cos)  # (max_seq_len, d_head//2)
        self.register_buffer('sin', sin)  # (max_seq_len, d_head//2)
        
    def forward(self, input_ids, past_kvs=None, mask=None):
        """
        Args:
            input_ids: (batch, seq_len) 输入的 token IDs
            past_kvs: list of past_kv tuples for each layer, 或 None
                      每个元素: (past_k, past_v)
                      past_k: (batch, past_seq_len, num_kv_heads, d_head)
                      past_v: (batch, past_seq_len, num_kv_heads, d_head)
            mask: (batch, seq_len, seq_len) 或 None
        
        Returns:
            logits: (batch, seq_len, vocab_size)
            new_kvs: list of new_kv tuples for each layer
        """
        batch, seq_len = input_ids.size()
        
        x = self.embed(input_ids)  # (batch, seq_len, dim)

        if past_kvs is None:
            past_kvs = [None] * self.num_layers
        
        new_kvs = []
        for layer, past_kv in zip(self.layers, past_kvs):
            x, new_kv = layer(
                x, 
                cos=self.cos, 
                sin=self.sin, 
                past_kv=past_kv, 
                mask=mask
            )
            new_kvs.append(new_kv)
        
        # 3. Final Norm + LM Head
        x = self.final_norm(x)
        logits = self.lm_head(x)  # (batch, seq_len, vocab_size)
        
        return logits, new_kvs
    
    @torch.no_grad()
    def generate(self, prompt, max_new_tokens=50, temperature=1.0, top_p=None,
                 use_cache=True):
        """自回归生成"""
        if use_cache:
            # KV Cache 模式：每步只传 1 个新 token
            logits, past_kvs = self.forward(prompt)
            next_token = self._sample(logits[:, -1], temperature, top_p)
            generated = [next_token]
            for _ in range(max_new_tokens - 1):
                logits, past_kvs = self.forward(next_token, past_kvs)
                next_token = self._sample(logits[:, -1], temperature, top_p)
                generated.append(next_token)
            return torch.cat([prompt] + generated, dim=1)
        else:
            # 无缓存模式：每步传完整序列（O(n²)）
            full_seq = prompt
            for _ in range(max_new_tokens):
                logits, _ = self.forward(full_seq)
                next_token = self._sample(logits[:, -1], temperature, top_p)
                full_seq = torch.cat([full_seq, next_token], dim=1)
            return full_seq

    def top_p_filter(self, logits, top_p=0.9):
        """top-p nucleus filtering — 至少保留概率最高的 token"""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        # 累积超过 top_p 的删掉，但第一个越界的保留（右移一位）
        mask = cumulative_probs > top_p
        mask[..., 1:] = mask[..., :-1].clone()
        mask[..., 0] = False
        sorted_logits[mask] = float('-inf')
        return sorted_logits.scatter(-1, sorted_indices, sorted_logits)

    def _sample(self, logits, temperature, top_p=None):
        # 清理异常值
        logits = torch.nan_to_num(logits, nan=-1e4, posinf=1e4, neginf=-1e4)
        if temperature > 0:
            logits = logits / temperature
        if top_p is not None and top_p < 1.0:
            logits = self.top_p_filter(logits, top_p)
        if temperature == 0:
            return logits.argmax(dim=-1, keepdim=True)
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, 1)