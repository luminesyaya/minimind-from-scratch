"""SFT LoRA 微调入口"""
import json
import os
import argparse
import torch
from core.models.transformer import MiniLLM
from core.models.lora import apply_lora_to_model, get_lora_params
from core.training.sft_dataset import load_sft_data, get_sft_batch
from core.training.loss import compute_loss
from core.training.optimizer import create_optimizer
from core.training.scheduler import create_scheduler
from core.training.checkpoint import save_ckpt


def main(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    model_cfg = config['model']
    train_cfg = config['training']
    data_cfg = config['data']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with open(data_cfg['meta_path'], 'r', encoding='utf-8') as f:
        meta = json.load(f)
    train_shape = (meta['n_train'], 2, meta['seq_len'])
    valid_shape = (meta['n_valid'], 2, meta['seq_len'])

    print(f"加载训练数据: {data_cfg['train_path']}")
    train_data = load_sft_data(data_cfg['train_path'], train_shape)
    print(f"训练集: {train_data.shape[0]} 条样本")

    valid_data = load_sft_data(data_cfg['valid_path'], valid_shape)
    print(f"验证集: {valid_data.shape[0]} 条样本")

    # 初始化模型 → 加载预训练权重 → 注入 LoRA
    model = MiniLLM(**model_cfg).to(device)

    pretrain_ckpt = train_cfg['pretrain_ckpt']
    print(f"加载预训练权重: {pretrain_ckpt}")
    ckpt = torch.load(pretrain_ckpt, map_location='cpu')
    model.load_state_dict(ckpt['model'])

    # 注入 LoRA
    apply_lora_to_model(model, r=train_cfg['lora_r'], alpha=train_cfg['lora_alpha'])

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"可训练参数: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    opt = create_optimizer(get_lora_params(model), train_cfg['lr'], train_cfg['weight_decay'])
    sched = create_scheduler(opt, train_cfg['warmup_steps'], train_cfg['max_steps'],
                             train_cfg['min_lr_ratio'])
    scaler = torch.cuda.amp.GradScaler()

    os.makedirs(train_cfg['out_dir'], exist_ok=True)

    print(f"\n设备: {device}")
    print(f"总步数: {train_cfg['max_steps']}, warmup: {train_cfg['warmup_steps']}")
    print(f"学习率: {train_cfg['lr']}, r: {train_cfg['lora_r']}, alpha: {train_cfg['lora_alpha']}\n")

    # 训练循环
    model.train()
    total_loss = 0.0

    for step in range(1, train_cfg['max_steps'] + 1):
        x, y = get_sft_batch(train_data, train_cfg['batch_size'], device)

        with torch.cuda.amp.autocast():
            logits, _ = model(x)
            loss = compute_loss(logits, y)

        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(get_lora_params(model), train_cfg['grad_clip'])
        scaler.step(opt)
        scaler.update()
        sched.step()

        total_loss += loss.item()

        if step % train_cfg['log_every'] == 0:
            avg = total_loss / train_cfg['log_every']
            lr = sched.get_last_lr()[0]
            print(f"Step {step:5d}/{train_cfg['max_steps']} | loss {avg:.4f} | lr {lr:.2e}")
            total_loss = 0.0

        if step % train_cfg['eval_every'] == 0:
            model.eval()
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    vx, vy = get_sft_batch(valid_data, train_cfg['batch_size'], device)
                    v_logits, _ = model(vx)
                    v_loss = compute_loss(v_logits, vy)
                    print(f"  [Eval] step {step} | val_loss {v_loss.item():.4f}")
            model.train()

        if step % train_cfg['save_every'] == 0:
            save_ckpt(model, opt, step,
                      os.path.join(train_cfg['out_dir'], f'ckpt_{step}.pt'))

    save_ckpt(model, opt, train_cfg['max_steps'],
              os.path.join(train_cfg['out_dir'], 'final.pt'))
    print(f"\n完成。模型保存在: {train_cfg['out_dir']}/final.pt")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/sft_lora.json')
    args = parser.parse_args()
    main(args.config)
