#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BIO 标签转换器 (jsonl_to_bio.py)

功能：
    读取 JSONL 格式的原始数据，将文本拆分为单字（字符级 token），
    并根据实体列表生成对应的 BIO 标签序列。

BIO 标注规范：
    - O: 非实体（Outside）
    - B-PER: 人名实体的开始（Beginning）
    - I-PER: 人名实体的内部（Inside）

输入格式：
    {"text": "张总您好", "entities": ["张"]}

输出格式：
    {"tokens": ["张", "总", "您", "好"], "labels": ["B-PER", "O", "O", "O"]}

使用方法：
    python jsonl_to_bio.py --input raw_ai_data.jsonl --output bio_data.jsonl

边界情况处理：
    - 实体重叠：保留先出现的实体，忽略重叠部分
    - 无实体：所有标签为 O
    - 实体不存在于文本中：跳过该实体
    - 多次出现的实体：标注所有出现位置
"""

import argparse
import json
from typing import Dict, List, Set, Tuple

# ============================================================
# 常量定义
# ============================================================

LABEL_O = "O"
LABEL_B_PER = "B-PER"
LABEL_I_PER = "I-PER"

# 复姓列表（用于辅助判断）
COMPOUND_SURNAMES = {
	"欧阳", "司马", "诸葛", "上官", "皇甫", "慕容", "东方", "令狐",
	"独孤", "轩辕", "公孙", "长孙", "宇文", "司徒", "司空", "夏侯",
	"太史", "闻人", "澹台", "淳于", "单于", "万俟", "申屠", "钟离"
}


def tokenize(text: str) -> List[str]:
	"""
	将文本拆分为字符级 token 列表。

	对于中文，每个字符作为一个 token；
	对于连续的英文字母/数字，作为一个整体 token。

	Args:
		text: 输入文本

	Returns:
		token 列表
	"""
	tokens = []
	current_token = ""
	is_english_mode = False

	for char in text:
		# 判断是否为英文字母、数字或常见符号
		is_english_char = char.isascii() and (char.isalnum() or char in "'-")

		if is_english_char:
			if not is_english_mode:
				# 切换到英文模式，保存之前的 token
				if current_token:
					tokens.append(current_token)
				current_token = char
				is_english_mode = True
			else:
				# 继续英文 token
				current_token += char
		else:
			if is_english_mode:
				# 切换到中文模式，保存英文 token
				if current_token:
					tokens.append(current_token)
				current_token = ""
				is_english_mode = False
			# 中文每个字符单独作为 token
			tokens.append(char)

	# 处理末尾可能剩余的 token
	if current_token:
		tokens.append(current_token)

	return tokens


def find_entity_positions(text: str, entity: str) -> List[Tuple[int, int]]:
	"""
	在文本中查找实体的所有出现位置。

	Args:
		text: 原始文本
		entity: 实体字符串

	Returns:
		位置列表，每个元素为 (start, end) 元组
	"""
	positions = []
	start = 0

	while True:
		pos = text.find(entity, start)
		if pos == -1:
			break
		positions.append((pos, pos + len(entity)))
		start = pos + 1  # 允许重叠匹配

	return positions


def positions_to_token_indices(
		text: str,
		tokens: List[str],
		positions: List[Tuple[int, int]]
) -> List[Tuple[int, int]]:
	"""
	将字符位置转换为 token 索引位置。

	Args:
		text: 原始文本
		tokens: token 列表
		positions: 字符位置列表

	Returns:
		token 索引列表，每个元素为 (start_idx, end_idx) 元组
	"""
	# 构建字符到 token 的映射
	char_to_token = {}
	char_idx = 0

	for token_idx, token in enumerate(tokens):
		for i in range(len(token)):
			if char_idx < len(text):
				char_to_token[char_idx] = token_idx
				char_idx += 1

	token_positions = []

	for start, end in positions:
		# 找到起始 token 索引
		start_token = char_to_token.get(start)
		# 找到结束 token 索引（end 是开区间）
		end_token = char_to_token.get(end - 1, start_token)

		if start_token is not None:
			token_positions.append((start_token, end_token + 1))

	return token_positions


def resolve_overlapping_entities(
		entity_spans: List[Tuple[int, int, str]]
) -> List[Tuple[int, int, str]]:
	"""
	解决实体重叠问题，采用"先到先得"策略。

	Args:
		entity_spans: 实体范围列表，每个元素为 (start, end, entity)

	Returns:
		无重叠的实体范围列表
	"""
	if not entity_spans:
		return []

	# 按起始位置排序，相同起始位置按长度排序（优先长实体）
	sorted_spans = sorted(entity_spans, key=lambda x: (x[0], -(x[1] - x[0])))

	result = []
	occupied: Set[int] = set()  # 已被占用的 token 索引

	for start, end, entity in sorted_spans:
		# 检查是否有重叠
		span_indices = set(range(start, end))
		if not span_indices.intersection(occupied):
			# 无重叠，添加到结果
			result.append((start, end, entity))
			occupied.update(span_indices)

	return result


def generate_bio_labels(
		tokens: List[str],
		entities: List[str],
		text: str
) -> List[str]:
	"""
	生成 BIO 标签序列。

	Args:
		tokens: token 列表
		entities: 实体列表
		text: 原始文本

	Returns:
		BIO 标签列表
	"""
	# 初始化所有标签为 O
	labels = [LABEL_O] * len(tokens)

	# 收集所有实体的位置
	all_spans = []

	for entity in entities:
		# 查找实体在文本中的位置
		char_positions = find_entity_positions(text, entity)
		if not char_positions:
			continue

		# 转换为 token 索引
		token_positions = positions_to_token_indices(text, tokens, char_positions)

		for start, end in token_positions:
			all_spans.append((start, end, entity))

	# 解决重叠
	resolved_spans = resolve_overlapping_entities(all_spans)

	# 生成标签
	for start, end, entity in resolved_spans:
		if start < len(tokens):
			labels[start] = LABEL_B_PER
			for i in range(start + 1, min(end, len(tokens))):
				labels[i] = LABEL_I_PER

	return labels


def convert_jsonl_to_bio(
		input_path: str,
		output_path: str,
		verbose: bool = False
) -> Dict[str, int]:
	"""
	转换 JSONL 文件为 BIO 格式。

	Args:
		input_path: 输入文件路径
		output_path: 输出文件路径
		verbose: 是否输出详细信息

	Returns:
		统计信息字典
	"""
	stats = {
		"total": 0,
		"with_entity": 0,
		"no_entity": 0,
		"skipped": 0,
		"total_entities": 0
	}

	with open(input_path, "r", encoding="utf-8") as infile, \
			open(output_path, "w", encoding="utf-8") as outfile:

		for line_num, line in enumerate(infile, 1):
			line = line.strip()
			if not line:
				continue

			try:
				data = json.loads(line)
				text = data.get("text", "")
				entities = data.get("entities", [])

				# 确保 entities 是列表
				if isinstance(entities, str):
					entities = [entities] if entities else []
				elif not isinstance(entities, list):
					entities = []

				# 过滤空实体
				entities = [e for e in entities if e and isinstance(e, str)]

				# 分词
				tokens = tokenize(text)

				if not tokens:
					stats["skipped"] += 1
					continue

				# 生成标签
				labels = generate_bio_labels(tokens, entities, text)

				# 验证长度一致
				if len(tokens) != len(labels):
					print(f"[警告] 行 {line_num}: tokens 和 labels 长度不一致，跳过")
					stats["skipped"] += 1
					continue

				# 写入输出
				output_item = {
					"tokens": tokens,
					"labels": labels
				}
				outfile.write(json.dumps(output_item, ensure_ascii=False) + "\n")

				# 更新统计
				stats["total"] += 1
				if entities:
					stats["with_entity"] += 1
					stats["total_entities"] += len(entities)
				else:
					stats["no_entity"] += 1

				if verbose:
					print(f"[{line_num}] {text[:30]}...")
					print(f"    tokens: {tokens[:10]}...")
					print(f"    entities: {entities}")
					print(f"    labels: {labels[:10]}...")
					print()

			except json.JSONDecodeError as e:
				print(f"[错误] 行 {line_num}: JSON 解析失败 - {e}")
				stats["skipped"] += 1
			except Exception as e:
				print(f"[错误] 行 {line_num}: 处理失败 - {e}")
				stats["skipped"] += 1

	return stats


def print_statistics(stats: Dict[str, int]) -> None:
	"""打印统计信息。"""
	print("\n" + "=" * 50)
	print("转换统计")
	print("=" * 50)
	print(f"总处理条数:     {stats['total']}")
	print(f"包含实体条数:   {stats['with_entity']}")
	print(f"无实体条数:     {stats['no_entity']}")
	print(f"跳过条数:       {stats['skipped']}")
	print(f"实体总数:       {stats['total_entities']}")

	if stats['total'] > 0:
		entity_ratio = stats['with_entity'] / stats['total'] * 100
		print(f"实体覆盖率:     {entity_ratio:.1f}%")
	print("=" * 50)


def main():
	"""主函数入口。"""
	parser = argparse.ArgumentParser(
		description="JSONL 到 BIO 标签格式转换器",
		formatter_class=argparse.RawDescriptionHelpFormatter,
		epilog="""
示例:
    python jsonl_to_bio.py --input raw_ai_data.jsonl --output bio_data.jsonl
    python jsonl_to_bio.py -i raw_ai_data.jsonl -o bio_data.jsonl -v

输入格式:
    {"text": "张总您好", "entities": ["张"]}

输出格式:
    {"tokens": ["张", "总", "您", "好"], "labels": ["B-PER", "O", "O", "O"]}
        """
	)

	parser.add_argument(
		"--input", "-i",
		type=str,
		default="raw_ai_data.jsonl",
		help="输入 JSONL 文件路径"
	)

	parser.add_argument(
		"--output", "-o",
		type=str,
		default="bio_data.jsonl",
		help="输出 JSONL 文件路径"
	)

	parser.add_argument(
		"--verbose", "-v",
		action="store_true",
		help="输出详细处理信息"
	)

	args = parser.parse_args()

	print(f"[信息] 输入文件: {args.input}")
	print(f"[信息] 输出文件: {args.output}")
	print("-" * 50)

	# 执行转换
	stats = convert_jsonl_to_bio(
		input_path=args.input,
		output_path=args.output,
		verbose=args.verbose
	)

	# 打印统计
	print_statistics(stats)


if __name__ == "__main__":
	main()
