from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from tokenizers.processors import TemplateProcessing
from tokenizers import decoders

# 1. 初始化 byte-level BPE
tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))

# 2. 预测分词：byte-level（GPT-2 风格）
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tokenizer.decoder = decoders.ByteLevel()

# 3. 特殊 token
special_tokens = ["<|endoftext|>", "<unk>", "<pad>"]

# 4. 训练参数
trainer = trainers.BpeTrainer(
    vocab_size=6400,
    special_tokens=special_tokens,
    min_frequency=2,
)

# 5. 训练 tokenizer
tokenizer.train(["data/pretrain_clean/train_subset.txt"], trainer)

# 6. 后处理
tokenizer.post_processor = TemplateProcessing(
    single="$A",
    special_tokens=[("<|endoftext|>", tokenizer.token_to_id("<|endoftext|>"))],
)

# 7. 保存 tokenizer
tokenizer.save("data/pretrain_clean/tokenizer.json")
print("Tokenizer saved to data/pretrain_clean/tokenizer.json")
print(f"Vocab size: {tokenizer.get_vocab_size()}")

# 8. 测试 tokenizer
test_texts = ["这是一条测试文本。", "你好世界。", "秋天的落叶飘落在微风中！"]
for t in test_texts:
    ids = tokenizer.encode(t).ids
    decoded = tokenizer.decode(ids)
    print(f"'{t}' -> {ids} -> '{decoded}'")