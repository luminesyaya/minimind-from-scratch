from torch.optim import AdamW

def create_optimizer(model, lr, weight_decay=0.1):
    return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95))