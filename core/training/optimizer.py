from torch.optim import AdamW

def create_optimizer(params_or_model, lr, weight_decay=0.1):
    if hasattr(params_or_model, 'parameters'):
        params = params_or_model.parameters()
    else:
        params = params_or_model
    return AdamW(params, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95))