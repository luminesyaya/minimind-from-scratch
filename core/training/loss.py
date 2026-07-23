import torch.nn.functional as F

def compute_loss(logits, labels):
    """
    logits: (batch, seq_len, vocab_size)
    labels: (batch, seq_len)
    """
    # shift: logits[:, :-1] → labels[:, 1:]
    logits = logits[:, :-1, :]  # (batch, seq_len-1, vocab_size)
    labels = labels[:, 1:]       # (batch, seq_len-1)
    
    # flatten → F.cross_entropy
    logits = logits.reshape(-1, logits.size(-1))  # (batch * (seq_len-1), vocab_size)
    labels = labels.reshape(-1)                    # (batch * (seq_len-1),)
    
    loss = F.cross_entropy(logits, labels)
    return loss