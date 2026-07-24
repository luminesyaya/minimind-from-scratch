import torch
import torch.nn.functional as F


def compute_logprob(logits, labels):
    """
    计算每个样本的平均对数概率（用于 DPO）
    
    Args:
        logits: (batch, seq_len, vocab_size) 模型输出的 logits
        labels: (batch, seq_len) 目标 token IDs
    
    Returns:
        mean_logp: (batch,) 每个样本的平均对数概率
        计算公式: mean_logp = (1/T) * Σ_t log p(y_t | y_<t)
    """
    # shift: 预测下一个 token
    logits = logits[:, :-1, :]  # (batch, seq_len-1, vocab_size)
    labels = labels[:, 1:]      # (batch, seq_len-1)
    
    # 计算每个位置的对数概率
    log_probs = F.log_softmax(logits, dim=-1)  # (batch, seq_len-1, vocab_size)
    
    gather_labels = labels.clone()
    gather_labels[gather_labels == -100] = 0
    
    # 取出 labels 对应位置的对数概率
    token_logps = log_probs.gather(dim=-1, index=gather_labels.unsqueeze(-1)).squeeze(-1)
    # token_logps: (batch, seq_len-1)
    
    # 计算每个样本的平均对数概率（忽略 -100）
    mask = (labels != -100).float()
    seq_len = mask.sum(dim=-1)  # (batch,)
    mean_logp = (token_logps * mask).sum(dim=-1) / (seq_len + 1e-8)
    # mean_logp: (batch,)
    
    return mean_logp


def dpo_loss(policy_chosen_logp, policy_rejected_logp,
             ref_chosen_logp, ref_rejected_logp, beta=0.1):
    """
    DPO (Direct Preference Optimization) 损失函数
    
    Args:
        policy_chosen_logp: (batch,) 当前模型对 chosen 回答的平均对数概率
        policy_rejected_logp: (batch,) 当前模型对 rejected 回答的平均对数概率
        ref_chosen_logp: (batch,) 参考模型对 chosen 回答的平均对数概率
        ref_rejected_logp: (batch,) 参考模型对 rejected 回答的平均对数概率
        beta: (float) 控制偏离参考模型程度的温度参数
    
    Returns:
        loss: (scalar) DPO 损失值
    
    Note:
        输入的 4 个值都是通过 compute_logprob() 计算得到的平均对数概率，
        每个值形状为 (batch,)，表示每个样本的 (1/T) * Σ_t log π(y_t | y_<t)
    
    DPO 公式:
        loss = -E[ log σ(β * (logπ(y_w|x) - logπ(y_l|x) - (logπ_ref(y_w|x) - logπ_ref(y_l|x))) ) ]
        其中 y_w 是 chosen, y_l 是 rejected
    """
    # 当前模型对 chosen 和 rejected 的概率差
    policy_ratio = policy_chosen_logp - policy_rejected_logp
    
    # 参考模型对 chosen 和 rejected 的概率差
    ref_ratio = ref_chosen_logp - ref_rejected_logp
    
    # 计算 logits（公式中的 β * (π_diff - π_ref_diff)）
    logits = beta * (policy_ratio - ref_ratio)
    
    # 返回负的 log-sigmoid 的均值
    return -F.logsigmoid(logits).mean()