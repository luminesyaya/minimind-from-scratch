"""
多轮对话交互脚本 - 加载 SFT 模型进行推理
"""
import json
import argparse
import torch
from tokenizers import Tokenizer
from core.models.transformer import MiniLLM


def render_prompt(history):
    """把多轮对话历史渲染成完整 prompt 文本"""
    text = ""
    for turn in history:
        text += f"### Human: {turn['user']}\n### Assistant: {turn['assistant']}\n"
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/sft.json')
    parser.add_argument('--ckpt', default='out/sft/final.pt')
    parser.add_argument('--tokenizer', default='data/pretrain_clean/tokenizer.json')
    parser.add_argument('--max-tokens', type=int, default=256)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top-p', type=float, default=None)
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        config = json.load(f)
    model_cfg = config['model']

    print(f"加载 tokenizer: {args.tokenizer}")
    tokenizer = Tokenizer.from_file(args.tokenizer)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"设备: {device}")

    model = MiniLLM(**model_cfg).to(device)

    print(f"加载模型: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    print(f"模型加载成功 (步数: {ckpt.get('step', 'unknown')})")

    print("\n" + "=" * 50)
    print("欢迎使用 MiniLLM Chat!")
    print("/clear 清空对话  /quit 退出")
    print("=" * 50 + "\n")

    model.eval()
    history = []           # [{"user": "...", "assistant": "..."}, ...]
    total_tokens = 0       # 已缓存历史的 token 数
    max_ctx = model_cfg['max_seq_len'] - args.max_tokens

    while True:
        try:
            user_input = input("你: ").strip()

            if user_input.lower() in ['quit', 'exit', '/quit']:
                print("再见！")
                break

            if user_input == '/clear':
                history = []
                total_tokens = 0
                print("对话历史已清空。\n")
                continue

            if not user_input:
                continue

            # 当前轮 prompt（只算历史 + 当前输入，不重复 encode 历史）
            current_turn = f"### Human: {user_input}\n### Assistant: "
            current_ids = tokenizer.encode(current_turn).ids
            cur_len = len(current_ids)

            # 超长裁剪：从最早轮次开始删，但永远保留 system prompt（history[0]）
            start = 0  # system 占位符，0 表示可以删第一轮
            while total_tokens + cur_len > max_ctx and len(history) > start:
                # 如果只剩 system prompt 了，停止裁剪
                removed_ids = tokenizer.encode(
                    f"### Human: {history[0]['user']}\n### Assistant: {history[0]['assistant']}\n"
                ).ids
                total_tokens -= len(removed_ids)
                if total_tokens < 0:
                    total_tokens = 0
                history.pop(0)

            # 组装完整 prompt：历史文本 + 当前 turn
            prompt = render_prompt(history) + current_turn
            input_ids = tokenizer.encode(prompt).ids
            input_ids = torch.tensor([input_ids], dtype=torch.long).to(device)

            # 生成
            with torch.no_grad():
                generated_ids = model.generate(
                    prompt=input_ids,
                    max_new_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p
                )

            # 解码
            prompt_len = input_ids.size(1)
            response_ids = generated_ids[0, prompt_len:].tolist()
            response = tokenizer.decode(response_ids)

            print(f"Assistant: {response}\n")

            # 存历史 + 更新 token 计数
            history.append({"user": user_input, "assistant": response})
            total_tokens += cur_len + len(response_ids)

        except KeyboardInterrupt:
            print("\n\n再见！")
            break
        except Exception as e:
            print(f"错误: {e}")
            continue


if __name__ == '__main__':
    main()
