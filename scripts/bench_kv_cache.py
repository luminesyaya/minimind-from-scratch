"""KV Cache Benchmark — 对比有缓存 vs 无缓存的生成速度"""
import json
import time
import torch
from tokenizers import Tokenizer
from core.models.transformer import MiniLLM

PROMPT_LENS = [16, 32, 64, 128, 256]
GEN_LENS = [32, 64, 128, 256]
WARMUP = 1
REPEAT = 3


def benchmark(model, prompt_ids, gen_len, use_cache, device):
    prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long).to(device)

    # 预热
    for _ in range(WARMUP):
        _ = model.generate(prompt_tensor, max_new_tokens=gen_len, temperature=0.0,
                           use_cache=use_cache)
    torch.cuda.synchronize()

    # 正式测量
    times = []
    for _ in range(REPEAT):
        prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long).to(device)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = model.generate(prompt_tensor, max_new_tokens=gen_len, temperature=0.0,
                           use_cache=use_cache)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    return sum(times) / len(times)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with open('configs/sft.json', 'r') as f:
        model_cfg = json.load(f)['model']

    tokenizer = Tokenizer.from_file('data/pretrain_clean/tokenizer.json')
    model = MiniLLM(**model_cfg).to(device)

    ckpt = torch.load('out/sft/final.pt', map_location='cpu')
    model.load_state_dict(ckpt['model'])
    model.eval()

    print(f"Device: {device}\n")
    print(f"{'Prompt':>8s} {'Gen':>6s} {'Cache(s)':>10s} {'NoCache(s)':>10s} {'Speedup':>8s}")
    print("-" * 48)

    results = []
    for p_len in PROMPT_LENS:
        prompt_ids = torch.randint(0, model_cfg['vocab_size'], (p_len,)).tolist()
        for g_len in GEN_LENS:
            t_cache = benchmark(model, prompt_ids, g_len, use_cache=True, device=device)
            t_nocache = benchmark(model, prompt_ids, g_len, use_cache=False, device=device)
            speedup = t_nocache / t_cache if t_cache > 0 else 0

            print(f"{p_len:>8d} {g_len:>6d} {t_cache:>10.4f} {t_nocache:>10.4f} {speedup:>7.2f}x")
            results.append((p_len, g_len, t_cache, t_nocache, speedup))

    # 统计
    speedups = [r[4] for r in results]
    avg = sum(speedups) / len(speedups)
    max_s = max(speedups)
    print(f"\n平均加速比: {avg:.2f}x  最大加速比: {max_s:.2f}x")


if __name__ == '__main__':
    main()
