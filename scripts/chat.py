"""
对话交互脚本 - 加载 SFT 模型进行推理
"""
import json
import os
import argparse
import torch
from tokenizers import Tokenizer
from core.models.transformer import MiniLLM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/sft.json', help='配置文件路径')
    parser.add_argument('--ckpt', default='out/sft/final.pt', help='模型 checkpoint 路径')
    parser.add_argument('--tokenizer', default='data/pretrain_clean/tokenizer.json', help='tokenizer 路径')
    parser.add_argument('--max-tokens', type=int, default=256, help='最大生成 token 数')
    parser.add_argument('--temperature', type=float, default=0.8, help='采样温度')
    args = parser.parse_args()
    
    # 1. 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = json.load(f)
    model_cfg = config['model']
    
    # 2. 加载 tokenizer
    print(f"加载 tokenizer: {args.tokenizer}")
    tokenizer = Tokenizer.from_file(args.tokenizer)
    
    # 3. 初始化模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"设备: {device}")
    
    model = MiniLLM(
        vocab_size=model_cfg['vocab_size'],
        dim=model_cfg['dim'],
        num_q_heads=model_cfg['num_q_heads'],
        num_kv_heads=model_cfg['num_kv_heads'],
        num_layers=model_cfg['num_layers'],
        max_seq_len=model_cfg['max_seq_len']
    )
    model.to(device)
    
    # 4. 加载 checkpoint
    print(f"加载模型: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    print(f"模型加载成功 (步数: {ckpt.get('step', 'unknown')})")
    
    # 5. 交互循环
    print("\n" + "="*50)
    print("欢迎使用 MiniLLM Chat!")
    print("输入 'quit' 或 'exit' 退出")
    print("="*50 + "\n")
    
    model.eval()
    
    while True:
        try:
            user_input = input("你: ").strip()
            
            if user_input.lower() in ['quit', 'exit']:
                print("再见！")
                break
            if not user_input:
                continue
            
            # 渲染 prompt
            prompt = f"### Human: {user_input}\n### Assistant: "
            
            # encode
            input_ids = tokenizer.encode(prompt).ids
            input_ids = torch.tensor([input_ids], dtype=torch.long).to(device)
            
            # 调用 model.generate() 生成
            with torch.no_grad():
                generated_ids = model.generate(
                    prompt=input_ids,
                    max_new_tokens=args.max_tokens,
                    temperature=args.temperature
                )
            
            # 解码：只取生成的部分（去掉 prompt）
            prompt_len = input_ids.size(1)
            response_ids = generated_ids[0, prompt_len:].tolist()
            response = tokenizer.decode(response_ids)
            
            print(f"Assistant: {response}\n")
            
        except KeyboardInterrupt:
            print("\n\n再见！")
            break
        except Exception as e:
            print(f"错误: {e}")
            continue


if __name__ == '__main__':
    main()