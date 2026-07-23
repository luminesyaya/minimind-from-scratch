from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

def create_scheduler(optimizer, warmup_steps, total_steps, min_lr_ratio=0.1):
    """
    warmup_steps 步内从 0.01*lr 线性升到 lr
    然后 cosine 降到 min_lr_ratio*lr
    """
    # 获取初始学习率
    lr = optimizer.param_groups[0]['lr']
    
    # Warmup: 从 0.01*lr 线性升到 lr
    warmup = LinearLR(
        optimizer, 
        start_factor=0.01, 
        end_factor=1.0, 
        total_iters=warmup_steps
    )
    
    # Cosine: 从 lr 降到 min_lr_ratio * lr
    cosine = CosineAnnealingLR(
        optimizer, 
        T_max=total_steps - warmup_steps, 
        eta_min=min_lr_ratio * lr
    )
    
    # Sequential: 先 warmup 再 cosine
    scheduler = SequentialLR(
        optimizer, 
        schedulers=[warmup, cosine], 
        milestones=[warmup_steps]
    )
    
    return scheduler