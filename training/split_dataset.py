#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据清洗与划分脚本
将原始数据清洗后划分为训练集、验证集、测试集
"""

import argparse
import json
import random
import re
from pathlib import Path
from collections import Counter


# ===================== 配置参数 =====================
RAW_DATA_DIR = Path(__file__).parent.parent / "raw_data"
PROCESSED_DATA_DIR = Path(__file__).parent.parent / "processed_data"

# 原始数据文件配置
RAW_DATA_FILES = {
    "bio_name_corpus": "bio_name_corpus.jsonl",
    "negative_samples": "negative_samples.jsonl",
    "training_data": "training_data.jsonl",
}

# 必须放入训练集的文件（不参与划分）
TRAIN_ONLY_FILES = {
    "training_data": "training_data.jsonl",
    "negative_samples": "negative_samples.jsonl",
}

# 划分比例
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

RANDOM_SEED = 42


# ===================== 数据验证函数 =====================
def validate_bio_item(item: dict) -> bool:
    """验证单条 BIO 数据的有效性"""
    tokens, labels = item.get("tokens", []), item.get("labels", [])
    
    # 基本检查
    if not tokens or not labels:
        return False
    if len(tokens) != len(labels):
        return False
    
    # 标签有效性检查
    valid_labels = {"O", "B-PER", "I-PER"}
    if any(l not in valid_labels for l in labels):
        return False
    
    # BIO 序列一致性检查：I-PER 必须跟在 B-PER 或 I-PER 后面
    for i, label in enumerate(labels):
        if label == "I-PER" and i > 0:
            prev_label = labels[i - 1]
            if prev_label not in {"B-PER", "I-PER"}:
                return False
    
    # 过滤无效文本
    text = "".join(tokens)
    if re.search(r'https?://|www\.|<[^>]+>', text, re.IGNORECASE):
        return False
    
    # 过滤纯标点符号
    punct_pattern = r'^[\s\u3000-\u303F\uFF00-\uFFEF\u2000-\u206F\u00A0-\u00BF!"#$%&\'()*+,\-./:;<=>?@\[\\\]^_`{|}~]+$'
    clean_tokens = [t for t in tokens if t.strip() and not re.match(punct_pattern, t)]
    
    return len(clean_tokens) >= 2


def check_entity_balance(data: list) -> dict:
    """检查数据集中实体的平衡性"""
    entity_counts = Counter()
    has_entity_count = 0
    
    for item in data:
        labels = item.get("labels", [])
        has_entity = False
        for label in labels:
            if label == "B-PER":
                entity_counts["B-PER"] += 1
                has_entity = True
            elif label == "I-PER":
                entity_counts["I-PER"] += 1
        if has_entity:
            has_entity_count += 1
    
    total = len(data)
    return {
        "total_samples": total,
        "samples_with_entity": has_entity_count,
        "samples_without_entity": total - has_entity_count,
        "entity_ratio": has_entity_count / total if total > 0 else 0,
        "label_distribution": dict(entity_counts),
    }


def clean_bio_data(data: list, verbose: bool = True) -> list:
    """清洗 BIO 数据"""
    original_count = len(data)
    cleaned = []
    invalid_reasons = Counter()
    
    for item in data:
        tokens, labels = item.get("tokens", []), item.get("labels", [])
        
        # 长度检查
        if len(tokens) < 3:
            invalid_reasons["too_short"] += 1
            continue
        
        if not validate_bio_item(item):
            invalid_reasons["invalid_format"] += 1
            continue
        
        cleaned.append(item)
    
    if verbose:
        print(f"  [清洗] 原始: {original_count} 条")
        print(f"  [清洗] 清洗后: {len(cleaned)} 条 (移除: {original_count - len(cleaned)})")
        if invalid_reasons:
            print(f"  [清洗] 移除原因: {dict(invalid_reasons)}")
    
    return cleaned


def load_jsonl_file(file_path: Path) -> list:
    """加载单个 JSONL 文件"""
    if not file_path.exists():
        print(f"  [警告] 文件不存在: {file_path}")
        return []
    
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  [警告] {file_path.name} 第 {line_num} 行 JSON 解析失败: {e}")
    
    print(f"  [加载] {file_path.name}: {len(data)} 条")
    return data


def save_jsonl(data: list, file_path: Path):
    """保存数据到 JSONL 文件"""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  [保存] {file_path.name}: {len(data)} 条")


def stratified_split(data: list, train_ratio: float, val_ratio: float, seed: int) -> tuple:
    """
    分层划分数据集，保持正负样本比例一致
    """
    random.seed(seed)
    
    # 分离有实体和无实体的样本
    has_entity = []
    no_entity = []
    
    for item in data:
        labels = item.get("labels", [])
        if "B-PER" in labels:
            has_entity.append(item)
        else:
            no_entity.append(item)
    
    random.shuffle(has_entity)
    random.shuffle(no_entity)
    
    def split_list(lst: list) -> tuple:
        n = len(lst)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        return lst[:train_end], lst[train_end:val_end], lst[val_end:]
    
    train_has, val_has, test_has = split_list(has_entity)
    train_no, val_no, test_no = split_list(no_entity)
    
    train_data = train_has + train_no
    val_data = val_has + val_no
    test_data = test_has + test_no
    
    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)
    
    return train_data, val_data, test_data


def main():
    parser = argparse.ArgumentParser(description="数据清洗与划分")
    parser.add_argument("--input-dir", type=str, default=str(RAW_DATA_DIR),
                        help="原始数据目录")
    parser.add_argument("--output-dir", type=str, default=str(PROCESSED_DATA_DIR),
                        help="输出目录")
    parser.add_argument("--train-ratio", type=float, default=TRAIN_RATIO,
                        help="训练集比例")
    parser.add_argument("--val-ratio", type=float, default=VAL_RATIO,
                        help="验证集比例")
    parser.add_argument("--test-ratio", type=float, default=TEST_RATIO,
                        help="测试集比例")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED,
                        help="随机种子")
    parser.add_argument("--no-split", action="store_true",
                        help="只清洗不划分")
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    print("=" * 60)
    print("数据清洗与划分")
    print("=" * 60)
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"划分比例: 训练={args.train_ratio}, 验证={args.val_ratio}, 测试={args.test_ratio}")
    print()
    
    # 1. 加载原始数据
    print("[步骤 1] 加载原始数据...")
    
    # 分离：必须放入训练集的数据 vs 可划分的数据
    train_only_data = []
    splittable_data = []
    
    for name, filename in RAW_DATA_FILES.items():
        file_path = input_dir / filename
        data = load_jsonl_file(file_path)
        
        if filename in TRAIN_ONLY_FILES.values():
            train_only_data.extend(data)
            print(f"  [标记] {filename} -> 必须放入训练集")
        else:
            splittable_data.extend(data)
    
    print(f"  [汇总] 必须训练数据: {len(train_only_data)} 条")
    print(f"  [汇总] 可划分数据: {len(splittable_data)} 条")
    print(f"  [汇总] 原始数据总计: {len(train_only_data) + len(splittable_data)} 条")
    print()
    
    # 2. 清洗数据
    print("[步骤 2] 清洗数据...")
    print("  [清洗] 必须训练数据:")
    cleaned_train_only = clean_bio_data(train_only_data)
    print("  [清洗] 可划分数据:")
    cleaned_splittable = clean_bio_data(splittable_data)
    
    # 显示清洗后的数据平衡性
    print("  [平衡] 必须训练数据:")
    train_only_balance = check_entity_balance(cleaned_train_only)
    print(f"    含实体样本: {train_only_balance['samples_with_entity']} ({train_only_balance['entity_ratio']:.1%})")
    print(f"    无实体样本: {train_only_balance['samples_without_entity']}")
    
    print("  [平衡] 可划分数据:")
    splittable_balance = check_entity_balance(cleaned_splittable)
    print(f"    含实体样本: {splittable_balance['samples_with_entity']} ({splittable_balance['entity_ratio']:.1%})")
    print(f"    无实体样本: {splittable_balance['samples_without_entity']}")
    print()
    
    if args.no_split:
        # 只清洗，不划分
        all_cleaned = cleaned_train_only + cleaned_splittable
        output_file = output_dir / "cleaned_data.jsonl"
        save_jsonl(all_cleaned, output_file)
        print("\n完成！数据已清洗并保存")
        return
    
    # 3. 划分数据集（只对可划分数据进行划分）
    print("[步骤 3] 划分数据集...")
    print("  [划分] 对可划分数据进行分层划分...")
    split_train, val_data, test_data = stratified_split(
        cleaned_splittable,
        args.train_ratio,
        args.val_ratio,
        args.seed
    )
    
    # 合并训练集：必须训练数据 + 划分得到的训练数据
    train_data = cleaned_train_only + split_train
    random.seed(args.seed)
    random.shuffle(train_data)
    
    print(f"  [划分] 可划分数据 -> 训练集: {len(split_train)} 条")
    print(f"  [划分] 可划分数据 -> 验证集: {len(val_data)} 条")
    print(f"  [划分] 可划分数据 -> 测试集: {len(test_data)} 条")
    print()
    print(f"  [合并] 必须训练数据: {len(cleaned_train_only)} 条")
    print(f"  [合并] 最终训练集: {len(train_data)} 条")
    print(f"  [合并] 最终验证集: {len(val_data)} 条")
    print(f"  [合并] 最终测试集: {len(test_data)} 条")
    print()
    
    # 4. 保存划分后的数据
    print("[步骤 4] 保存数据集...")
    save_jsonl(train_data, output_dir / "train.jsonl")
    save_jsonl(val_data, output_dir / "val.jsonl")
    save_jsonl(test_data, output_dir / "test.jsonl")
    
    # 5. 保存统计信息
    all_cleaned = cleaned_train_only + cleaned_splittable
    stats = {
        "total_samples": len(all_cleaned),
        "train_only_samples": len(cleaned_train_only),
        "splittable_samples": len(cleaned_splittable),
        "train_samples": len(train_data),
        "val_samples": len(val_data),
        "test_samples": len(test_data),
        "train_only_balance": train_only_balance,
        "splittable_balance": splittable_balance,
        "train_balance": check_entity_balance(train_data),
        "val_balance": check_entity_balance(val_data),
        "test_balance": check_entity_balance(test_data),
    }
    
    stats_file = output_dir / "stats.json"
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"  [保存] stats.json")
    
    print("\n" + "=" * 60)
    print("完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
