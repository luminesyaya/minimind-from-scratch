import json
import re
import hashlib
import argparse

def clean_control_chars(text):
    """
    去掉控制字符，保留\t,\n,\r
    """
    import re
    control_chars = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
    return control_chars.sub('', text)

def clean_html(text):
    """去掉HTML标签<...>"""
    import re 
    return re.sub(r'<[^>]+>', '', text)

def compress_whitespace(text):
    """连续空白压缩成单个"""
    import re
    # 连续空格/制表符 → 单个空格
    text = re.sub(r'[ \t]+', ' ', text)
    # 连续换行/回车 → 单个换行
    text = re.sub(r'[\r\n]+', '\n', text)
    return text.strip()

def filter_length(text, min_len, max_len):
    """长度过滤"""
    length = len(text)
    return min_len <= length <= max_len

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/pretrain_t2t_mini.jsonl')
    parser.add_argument('--output-dir', default='data/pretrain_clean')
    parser.add_argument('--clean-html', action='store_true')
    parser.add_argument('--compress-whitespace', action='store_true')
    parser.add_argument('--min-length', type=int, default=10)
    parser.add_argument('--max-length', type=int, default=10000)
    parser.add_argument('--dedup', action='store_true')
    args = parser.parse_args()
    
    # 第 1 步：读 JSONL + 提取 text
    docs = []
    with open(args.input, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if 'text' in obj:
                    docs.append(obj['text'])
                else:
                    print(f"行缺少 'text' 字段，跳过")
            except json.JSONDecodeError:
                print(f"无法解析JSON: {line[:50]}...，跳过")
                
        print(f"原始文档数: {len(docs)}")
        
    # 第 2 步：逐条清洗，每条规则打印命中数
    cleaned_docs = []
    total = len(docs)
    
     # 初始化计数器
    ctrl_hit = 0      # 控制字符命中数
    html_hit = 0      # HTML 标签命中数
    ws_hit = 0        # 空白压缩命中数
    len_drop = 0      # 长度过滤丢弃数
    
    for i, doc in enumerate(docs):
        origin = doc
        
        if args.clean_html:
            new_doc = clean_html(doc)
            if new_doc != doc:
                html_hit += 1
            doc = new_doc
        
        new_doc = clean_control_chars(doc)
        if new_doc != doc:
            ctrl_hit += 1
        doc = new_doc
        
        if args.compress_whitespace:
            new_doc = compress_whitespace(doc)
            if new_doc != doc:
                ws_hit += 1
            doc = new_doc
        
        if not filter_length(doc, args.min_length, args.max_length):
            len_drop += 1
            continue
        
        cleaned_docs.append(doc)
        
    print(f"\n 清洗统计:")
    print(f"   HTML 清理命中: {html_hit} 条")
    print(f"   控制字符清理命中: {ctrl_hit} 条")
    print(f"   空白压缩命中: {ws_hit} 条")
    print(f"   长度过滤丢弃: {len_drop} 条")
    print(f"   最终保留: {len(cleaned_docs)} / {total} 条")
    
    # 第 3 步：SHA256 去重
    docs = cleaned_docs
    if args.dedup:
        original_count = len(docs)
        seen = set()
        unique_docs = []
        dedup_hit = 0
        
        for doc in docs:
            # 计算 SHA256 哈希值
            doc_hash = hashlib.sha256(doc.encode('utf-8')).hexdigest()
            
            # 检查是否已存在
            if doc_hash in seen:
                dedup_hit += 1
                continue
            
            seen.add(doc_hash)
            unique_docs.append(doc)
        
        docs = unique_docs
        print(f"\n 去重统计:")
        print(f"    去重前文档数: {original_count}")
        print(f"    去重后文档数: {len(docs)}")
        print(f"    去重命中数: {dedup_hit}")
    else:
        print(f"\n 跳过去重（--dedup 未启用）")
        
    # 第 4 步：SHA1 切分 99:1 训练集/验证集
    train_docs = []
    val_docs = []
    
    for doc in docs:
        # 计算 SHA1 哈希值，取前 8 位转整数
        hash_hex = hashlib.sha1(doc.encode('utf-8')).hexdigest()[:8]
        hash_int = int(hash_hex, 16)  # 16 进制转 10 进制
        partition = hash_int % 100    # 取模 100，得到 0-99
        
        if partition < 99:
            train_docs.append(doc)
        else:  # partition == 99
            val_docs.append(doc)
        
    print(f"\n 数据集切分统计:")
    print(f"   训练集: {len(train_docs)} 条 ({len(train_docs)/len(docs)*100:.1f}%)")
    print(f"   验证集: {len(val_docs)} 条 ({len(val_docs)/len(docs)*100:.1f}%)")
    
    # 第 5 步：写出 txt，文档间加 <|endoftext|>
    
    # 保存训练集
    with open('train.txt', 'w', encoding='utf-8') as f:
        for doc in train_docs:
            f.write(doc + '<|endoftext|>\n')
    print(f" 训练集已保存: train.txt ({len(train_docs)} 条)")
        
    # 保存验证集
    with open('valid.txt', 'w', encoding='utf-8') as f:
        for doc in val_docs:
            f.write(doc + '<|endoftext|>\n')
    print(f" 验证集已保存: valid.txt ({len(val_docs)} 条)")
    
    # 可选：保存统计信息到 meta.json
    meta = {
        'total_docs': len(docs),
        'train_count': len(train_docs),
        'valid_count': len(val_docs),
        'min_length': args.min_length,
        'max_length': args.max_length,
        'clean_html': args.clean_html,
        'compress_whitespace': args.compress_whitespace,
        'dedup': args.dedup,
    }
    
    
    with open('meta.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f" 元信息已保存: meta.json")
    
if __name__ == '__main__':
    main()