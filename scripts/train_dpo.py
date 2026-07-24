"""DPO 训练入口 — 直接偏好优化"""
import json
import os
import argparse
import torch
from core.models.transformer import MiniLLM
from core.training.dpo_dataset import load_dpo_data, get_dpo_batch
from core.training.dpo_loss import compute_logprob, dpo_loss
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

    # 加载数据形状
    with open(data_cfg['meta_path'], 'r', encoding='utf-8') as f:
        meta = json.load(f)
    train_shape = (meta['n_train'], 2, meta['seq_len'])
    valid_shape = (meta['n_valid'], 2, meta['seq_len'])

    print(f"加载训练数据: {data_cfg['train_path']}")
    train_data = load_dpo_data(data_cfg['train_path'], train_shape)
    print(f"训练集: {train_data.shape[0]} 条样本")

    valid_data = load_dpo_data(data_cfg['valid_path'], valid_shape)
    print(f"验证集: {valid_data.shape[0]} 条样本")

    # Reference 模型（SFT checkpoint，冻结）
    print(f"加载 Reference 模型: {train_cfg['ref_ckpt']}")
    ref_model = MiniLLM(**model_cfg).to(device)
    ref_ckpt = torch.load(train_cfg['ref_ckpt'], map_location='cpu')
    ref_model.load_state_dict(ref_ckpt['model'])
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    # Policy 模型（从 ref 初始化，要训练）
    policy_model = MiniLLM(**model_cfg).to(device)
    policy_model.load_state_dict(ref_ckpt['model'])

    print(f"Params: {sum(p.numel() for p in policy_model.parameters()):,}")

    # 优化器 + 调度器
    opt = create_optimizer(policy_model, train_cfg['lr'], train_cfg['weight_decay'])
    sched = create_scheduler(opt, train_cfg['warmup_steps'], train_cfg['max_steps'],
                             train_cfg['min_lr_ratio'])
    scaler = torch.cuda.amp.GradScaler()

    os.makedirs(train_cfg['out_dir'], exist_ok=True)

    print(f"\n设备: {device}")
    print(f"总步数: {train_cfg['max_steps']}, warmup: {train_cfg['warmup_steps']}")
    print(f"学习率: {train_cfg['lr']}, beta: {train_cfg['beta']}")
    print()

    # 训练循环
    policy_model.train()
    total_loss = 0.0

    for step in range(1, train_cfg['max_steps'] + 1):
        c_ids, c_labels, r_ids, r_labels = get_dpo_batch(
            train_data, train_cfg['batch_size'], device
        )

        with torch.cuda.amp.autocast():
            p_c_logits, _ = policy_model(c_ids)
            p_r_logits, _ = policy_model(r_ids)

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                r_c_logits, _ = ref_model(c_ids)
                r_r_logits, _ = ref_model(r_ids)

        p_c_lp = compute_logprob(p_c_logits, c_labels)
        p_r_lp = compute_logprob(p_r_logits, r_labels)
        r_c_lp = compute_logprob(r_c_logits, c_labels)
        r_r_lp = compute_logprob(r_r_logits, r_labels)

        loss = dpo_loss(p_c_lp, p_r_lp, r_c_lp, r_r_lp, beta=train_cfg['beta'])

        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(policy_model.parameters(), train_cfg['grad_clip'])
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
            policy_model.eval()
            with torch.no_grad():
                vc_ids, vc_labels, vr_ids, vr_labels = get_dpo_batch(
                    valid_data, train_cfg['batch_size'], device
                )
                with torch.cuda.amp.autocast():
                    vp_c_logits, _ = policy_model(vc_ids)
                    vp_r_logits, _ = policy_model(vr_ids)
                    vr_c_logits, _ = ref_model(vc_ids)
                    vr_r_logits, _ = ref_model(vr_ids)

                vp_c_lp = compute_logprob(vp_c_logits, vc_labels)
                vp_r_lp = compute_logprob(vp_r_logits, vr_labels)
                vr_c_lp = compute_logprob(vr_c_logits, vc_labels)
                vr_r_lp = compute_logprob(vr_r_logits, vr_labels)
                v_loss = dpo_loss(vp_c_lp, vp_r_lp, vr_c_lp, vr_r_lp, beta=train_cfg['beta'])
                print(f"  [Eval] step {step} | val_loss {v_loss.item():.4f}")
            policy_model.train()

        if step % train_cfg['save_every'] == 0:
            save_ckpt(policy_model, opt, step,
                      os.path.join(train_cfg['out_dir'], f'ckpt_{step}.pt'))

    save_ckpt(policy_model, opt, train_cfg['max_steps'],
              os.path.join(train_cfg['out_dir'], 'final.pt'))
    print(f"\n完成。模型保存在: {train_cfg['out_dir']}/final.pt")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/dpo.json')
    args = parser.parse_args()
    main(args.config)
