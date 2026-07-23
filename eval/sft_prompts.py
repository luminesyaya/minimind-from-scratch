"""固定 Prompt 评测脚本 — 对比 pretrain vs SFT 的对话能力"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import torch
from tokenizers import Tokenizer
from core.models.transformer import MiniLLM

PROMPTS = [
    "请用简单的话解释一下什么是机器学习",
    "如果我想保持身体健康，有什么建议",
    "写一首关于秋天的短诗",
    "请介绍一下中国的四大发明",
    "如何面对生活中的困难与挑战",
    "Python 语言有什么特点",
    "推荐三本值得阅读的书籍并说明理由",
    "你认为什么是幸福",
]


def load_model(ckpt_path, model_cfg, device):
    model = MiniLLM(
        vocab_size=model_cfg["vocab_size"],
        dim=model_cfg["dim"],
        num_q_heads=model_cfg["num_q_heads"],
        num_kv_heads=model_cfg["num_kv_heads"],
        num_layers=model_cfg["num_layers"],
        max_seq_len=model_cfg["max_seq_len"],
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


@torch.no_grad()
def evaluate_model(model, tokenizer, device, label):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    for i, prompt_text in enumerate(PROMPTS, 1):
        rendered = f"### Human: {prompt_text}\n### Assistant: "
        input_ids = tokenizer.encode(rendered).ids
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)

        max_new = min(256, 512 - len(input_ids))
        generated = model.generate(
            input_tensor, max_new_tokens=max_new, temperature=0.8
        )
        response_ids = generated[0, len(input_ids) :].tolist()
        response = tokenizer.decode(response_ids)

        print(f"\n[{i}] Q: {prompt_text}")
        print(f"    A: {response}")
        print(f"    --- 评分: __ / 5")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open("configs/sft.json", "r") as f:
        model_cfg = json.load(f)["model"]

    tokenizer = Tokenizer.from_file("data/pretrain_clean/tokenizer.json")
    print(f"Device: {device}")

    # pretrain
    evaluate_model(
        load_model("out/pretrain/ckpt.pt", model_cfg, device),
        tokenizer, device, "Pretrain Checkpoint"
    )

    # SFT
    evaluate_model(
        load_model("out/sft/final.pt", model_cfg, device),
        tokenizer, device, "SFT Checkpoint"
    )


if __name__ == "__main__":
    main()
