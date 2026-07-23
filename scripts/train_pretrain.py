import json, os, argparse, torch
import numpy as np
from core.models.transformer import MiniLLM
from core.training.dataset import load_token_data, get_batch
from core.training.loss import compute_loss
from core.training.optimizer import create_optimizer
from core.training.scheduler import create_scheduler
from core.training.checkpoint import save_ckpt, load_ckpt

def load_config(path):
    """读 configs/pretrain.json"""
    with open(path) as f:
        return json.load(f)
    
def main(config_path):
    cfg = load_config(config_path)
    model_cfg = cfg['model']
    train_cfg = cfg['training']
    data_cfg = cfg['data']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_data = load_token_data(data_cfg['train_path'])
    valid_data = load_token_data(data_cfg['valid_path'])
    print(f"Train tokens: {len(train_data):,}")
    
    model = MiniLLM(**model_cfg).to(device)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    
    opt = create_optimizer(model, train_cfg['lr'], train_cfg['weight_decay'])
    sched = create_scheduler(opt, train_cfg['warmup_steps'], train_cfg['max_steps'], train_cfg['min_lr_ratio'])
    scaler = torch.cuda.amp.GradScaler()
    
    ckpt_path = os.path.join(train_cfg.get('out_dir', 'out'), 'ckpt.pt')
    start_step = load_ckpt(ckpt_path, model, opt) if os.path.exists(ckpt_path) else 0
    
    for step in range(start_step, train_cfg['max_steps']):
        x, y = get_batch(train_data, train_cfg['batch_size'], train_cfg['seq_len'], device)
        with torch.cuda.amp.autocast():
            logits, _ = model(x)
            loss = compute_loss(logits, y)

        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg['grad_clip'])
        scaler.step(opt)
        scaler.update()
        sched.step()
        
        # 日志
        if step % train_cfg['log_every'] == 0 or step == train_cfg['max_steps'] - 1:
            lr = sched.get_last_lr()[0]
            print(f"Step {step:5d} | loss {loss.item():.4f} | lr {lr:.2e}")
            
        # 验证
        if step % train_cfg['eval_every'] == 0 and step > 0:
            model.eval()
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    vx, vy = get_batch(valid_data, train_cfg['batch_size'], train_cfg['seq_len'], device)
                    v_logits, _ = model(vx)
                    v_loss = compute_loss(v_logits, vy)
            print(f"  [Eval] step {step} | val_loss {v_loss.item():.4f}")
            model.train()
            
        # 保存
        if step % 1000 == 0 and step > 0:
            os.makedirs(train_cfg.get('out_dir', 'out'), exist_ok=True)
            save_ckpt(model, opt, step, ckpt_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/pretrain.json')
    args = parser.parse_args()
    main(args.config)