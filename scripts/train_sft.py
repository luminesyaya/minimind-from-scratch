"""SFT 微调入口"""
import json
import os
import argparse
import torch
import numpy as np
from core.models.transformer import MiniLLM
from core.training.sft_dataset import load_sft_data, get_sft_batch
from core.training.loss import compute_loss
from core.training.optimizer import create_optimizer
from core.training.scheduler import create_scheduler
from core.training.checkpoint import save_ckpt, load_ckpt


def main(config_path):
    # 1. 读配置文件
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 模型配置
    model_cfg = config['model']
    vocab_size = model_cfg['vocab_size']
    dim = model_cfg['dim']
    num_q_heads = model_cfg['num_q_heads']
    num_kv_heads = model_cfg['num_kv_heads']
    num_layers = model_cfg['num_layers']
    max_seq_len = model_cfg['max_seq_len']
    
    # 训练配置
    train_cfg = config['training']
    batch_size = train_cfg['batch_size']
    max_steps = train_cfg['max_steps']
    warmup_steps = train_cfg['warmup_steps']
    lr = train_cfg['lr']
    min_lr_ratio = train_cfg['min_lr_ratio']
    weight_decay = train_cfg['weight_decay']
    grad_clip = train_cfg['grad_clip']
    log_every = train_cfg['log_every']
    eval_every = train_cfg['eval_every']
    save_every = train_cfg['save_every']
    out_dir = train_cfg['out_dir']
    pretrain_ckpt = train_cfg.get('pretrain_ckpt', None)
    
    # 数据配置
    data_cfg = config['data']
    train_path = data_cfg['train_path']
    valid_path = data_cfg['valid_path']
    meta_path = data_cfg['meta_path']
    
    # 从 meta.json 读取数据形状
    with open(meta_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    train_shape = tuple(meta['train_shape'])  # (n_samples, 2, seq_len)
    valid_shape = tuple(meta['valid_shape'])
    seq_len = train_shape[2]  # 从数据中获取 seq_len
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 2. 加载 SFT 数据
    print(f"加载训练数据: {train_path}")
    train_data = load_sft_data(train_path, train_shape)
    print(f"训练集: {train_data.shape[0]} 条样本, shape: {train_data.shape}")
    
    if valid_path and os.path.exists(valid_path):
        valid_data = load_sft_data(valid_path, valid_shape)
        print(f"验证集: {valid_data.shape[0]} 条样本, shape: {valid_data.shape}")
    else:
        valid_data = None
        print("没有验证集")
    
    # 3. 初始化模型 → 加载预训练 checkpoint
    model = MiniLLM(
        vocab_size=vocab_size,
        dim=dim,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        num_layers=num_layers,
        max_seq_len=max_seq_len
    )
    model.to(device)
    
    # 加载预训练权重（如果有）
    start_step = 0
    if pretrain_ckpt and os.path.exists(pretrain_ckpt):
        print(f"加载预训练权重: {pretrain_ckpt}")
        # 只加载模型权重，不加载优化器状态
        ckpt = torch.load(pretrain_ckpt, map_location='cpu')
        model.load_state_dict(ckpt['model'])
        print(f"预训练权重加载成功，步数: {ckpt.get('step', 0)}")
    elif pretrain_ckpt:
        print(f"警告: 预训练权重文件不存在 {pretrain_ckpt}")
    
    # 4. 优化器 + scheduler
    optimizer = create_optimizer(model, lr=lr, weight_decay=weight_decay)
    scheduler = create_scheduler(optimizer, warmup_steps, max_steps, min_lr_ratio)
    
    print(f"\n设备: {device}")
    print(f"总步数: {max_steps}, warmup: {warmup_steps}")
    print(f"学习率: {lr}, weight_decay: {weight_decay}")
    print(f"输出目录: {out_dir}\n")
    
    # 5. 训练循环
    os.makedirs(out_dir, exist_ok=True)
    model.train()
    
    global_step = 0
    total_loss = 0
    
    print("开始 SFT 训练...")
    for step in range(1, max_steps + 1):
        # 获取 batch
        x, y = get_sft_batch(train_data, batch_size, device)
        
        # 前向传播
        logits, _ = model(x)
        
        # 计算 loss (自动忽略 -100)
        loss = compute_loss(logits, y)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        global_step += 1
        
        # 打印日志
        if step % log_every == 0:
            avg_loss = total_loss / log_every
            current_lr = scheduler.get_last_lr()[0]
            print(f"Step {step}/{max_steps} | loss: {avg_loss:.4f} | lr: {current_lr:.2e}")
            total_loss = 0
        
        # 验证
        if step % eval_every == 0 and valid_data is not None:
            model.eval()
            with torch.no_grad():
                x_val, y_val = get_sft_batch(valid_data, batch_size, device)
                logits_val, _ = model(x_val)
                val_loss = compute_loss(logits_val, y_val)
                print(f"Step {step} | 验证 loss: {val_loss.item():.4f}")
            model.train()
        
        # 保存 checkpoint
        if step % save_every == 0:
            ckpt_path = os.path.join(out_dir, f'ckpt_{step}.pt')
            save_ckpt(model, optimizer, step, ckpt_path)
            print(f"Checkpoint 已保存: {ckpt_path}")
    
    # 保存最终模型
    final_path = os.path.join(out_dir, 'final.pt')
    save_ckpt(model, optimizer, global_step, final_path)
    print(f"\n训练完成! 最终模型已保存: {final_path}")
    print(f"总步数: {global_step}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/sft.json')
    args = parser.parse_args()
    main(args.config)