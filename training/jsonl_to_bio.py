#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import html
import json
import re
from typing import Dict, List, Optional, Set, Tuple

LABEL_O = "O"
LABEL_B_PER = "B-PER"
LABEL_I_PER = "I-PER"

COMPOUND_SURNAMES = {
	"欧阳", "司马", "诸葛", "上官", "皇甫", "慕容", "东方", "令狐",
	"独孤", "轩辕", "公孙", "长孙", "宇文", "司徒", "司空", "夏侯",
	"太史", "闻人", "澹台", "淳于", "单于", "万俟", "申屠", "钟离"
}


class TextCleaner:
    def __init__(self):
        self.url_pattern = re.compile(
            r'(?:https?://|www\.)[^\s<>"{}|\\^`\[\]]+'
            r'|'
            r'[a-zA-Z0-9][-a-zA-Z0-9]*\.(?:com|cn|net|org|edu|gov|io|co|info|biz|cc|tv|me|xyz|top|vip|shop|online|site|club)[^\s<>"{}|\\^`\[\]]*',
            re.IGNORECASE
        )
        self.html_tag_pattern = re.compile(
            r'<[^>]+>'
            r'|&[a-zA-Z]+;'
            r'|&#\d+;'
            r'|&#x[0-9a-fA-F]+;'
        )
        self.control_char_pattern = re.compile(
            r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]'
            r'|[\u200b-\u200f\u2028-\u202f\u205f-\u206f\ufeff]'
        )
        self.order_number_pattern = re.compile(
            r'\b(?:订单号?|单号|编号|流水号|交易号|运单号)[:：\s]*([A-Za-z0-9]{8,})\b'
            r'|\b[A-Z]{2,4}\d{8,}\b'
            r'|\b\d{12,}\b'
        )

    def clean(self, text: str) -> str:
        text = html.unescape(text)
        text = self.html_tag_pattern.sub('', text)
        text = self.url_pattern.sub('', text)
        text = self.order_number_pattern.sub('', text)
        text = self.control_char_pattern.sub('', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def update_entities(self, original_text: str, cleaned_text: str, entities: List[str]) -> List[str]:
        return [e for e in entities if e and e in cleaned_text]


def tokenize(text: str) -> List[str]:
	tokens = []
	current_token = ""
	is_english_mode = False

	for char in text:
		is_english_char = char.isascii() and (char.isalnum() or char in "'-")

		if is_english_char:
			if not is_english_mode:
				if current_token:
					tokens.append(current_token)
				current_token = char
				is_english_mode = True
			else:
				current_token += char
		else:
			if is_english_mode:
				if current_token:
					tokens.append(current_token)
				current_token = ""
				is_english_mode = False
			tokens.append(char)

	if current_token:
		tokens.append(current_token)

	return tokens


def find_entity_positions(text: str, entity: str) -> List[Tuple[int, int]]:
	positions = []
	start = 0

	while True:
		pos = text.find(entity, start)
		if pos == -1:
			break
		positions.append((pos, pos + len(entity)))
		start = pos + 1

	return positions


def positions_to_token_indices(
		text: str,
		tokens: List[str],
		positions: List[Tuple[int, int]]
) -> List[Tuple[int, int]]:
	char_to_token = {}
	char_idx = 0

	for token_idx, token in enumerate(tokens):
		for i in range(len(token)):
			if char_idx < len(text):
				char_to_token[char_idx] = token_idx
				char_idx += 1

	token_positions = []

	for start, end in positions:
		start_token = char_to_token.get(start)
		end_token = char_to_token.get(end - 1, start_token)

		if start_token is not None:
			token_positions.append((start_token, end_token + 1))

	return token_positions


def resolve_overlapping_entities(
		entity_spans: List[Tuple[int, int, str]]
) -> List[Tuple[int, int, str]]:
	if not entity_spans:
		return []

	sorted_spans = sorted(entity_spans, key=lambda x: (x[0], -(x[1] - x[0])))

	result = []
	occupied: Set[int] = set()

	for start, end, entity in sorted_spans:
		span_indices = set(range(start, end))
		if not span_indices.intersection(occupied):
			result.append((start, end, entity))
			occupied.update(span_indices)

	return result


def generate_bio_labels(
		tokens: List[str],
		entities: List[str],
		text: str
) -> List[str]:
	labels = [LABEL_O] * len(tokens)

	all_spans = []

	for entity in entities:
		char_positions = find_entity_positions(text, entity)
		if not char_positions:
			continue

		token_positions = positions_to_token_indices(text, tokens, char_positions)

		for start, end in token_positions:
			all_spans.append((start, end, entity))

	resolved_spans = resolve_overlapping_entities(all_spans)

	for start, end, entity in resolved_spans:
		if start < len(tokens):
			labels[start] = LABEL_B_PER
			for i in range(start + 1, min(end, len(tokens))):
				labels[i] = LABEL_I_PER

	return labels


def convert_jsonl_to_bio(
		input_path: str,
		output_path: str,
		verbose: bool = False,
		clean: bool = False
) -> Dict[str, int]:
	stats = {
		"total": 0,
		"with_entity": 0,
		"no_entity": 0,
		"skipped": 0,
		"total_entities": 0,
		"cleaned_items": 0,
		"entities_after_clean": 0
	}

	cleaner = TextCleaner() if clean else None

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

				if isinstance(entities, str):
					entities = [entities] if entities else []
				elif not isinstance(entities, list):
					entities = []

				entities = [e for e in entities if e and isinstance(e, str)]

				if cleaner:
					original_text = text
					original_entities = entities.copy()
					text = cleaner.clean(text)
					if text != original_text:
						stats["cleaned_items"] += 1
					entities = cleaner.update_entities(original_text, text, entities)
					stats["entities_after_clean"] += len(entities)

				if not text or len(text.strip()) < 2:
					stats["skipped"] += 1
					continue

				tokens = tokenize(text)

				if not tokens:
					stats["skipped"] += 1
					continue

				labels = generate_bio_labels(tokens, entities, text)

				if len(tokens) != len(labels):
					print(f"[警告] 行 {line_num}: tokens 和 labels 长度不一致，跳过")
					stats["skipped"] += 1
					continue

				output_item = {
					"tokens": tokens,
					"labels": labels
				}
				outfile.write(json.dumps(output_item, ensure_ascii=False) + "\n")

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


def print_statistics(stats: Dict[str, int], clean: bool = False) -> None:
	print("\n" + "=" * 50)
	print("转换统计")
	print("=" * 50)
	print(f"总处理条数:     {stats['total']}")
	print(f"包含实体条数:   {stats['with_entity']}")
	print(f"无实体条数:     {stats['no_entity']}")
	print(f"跳过条数:       {stats['skipped']}")
	print(f"实体总数:       {stats['total_entities']}")

	if clean and stats.get('cleaned_items', 0) > 0:
		print("-" * 50)
		print("清洗统计:")
		print(f"  清洗条数:     {stats['cleaned_items']}")
		print(f"  清洗后实体:   {stats['entities_after_clean']}")

	if stats['total'] > 0:
		entity_ratio = stats['with_entity'] / stats['total'] * 100
		print(f"实体覆盖率:     {entity_ratio:.1f}%")
	print("=" * 50)


def main():
	parser = argparse.ArgumentParser(description="JSONL 到 BIO 标签格式转换器")

	parser.add_argument("--input", "-i", type=str, default="raw_ai_data.jsonl", help="输入 JSONL 文件路径")
	parser.add_argument("--output", "-o", type=str, default="bio_data.jsonl", help="输出 JSONL 文件路径")
	parser.add_argument("--verbose", "-v", action="store_true", help="输出详细处理信息")
	parser.add_argument("--clean", "-c", action="store_true", help="转换前清洗数据")

	args = parser.parse_args()

	print(f"[信息] 输入文件: {args.input}")
	print(f"[信息] 输出文件: {args.output}")
	if args.clean:
		print("[信息] 数据清洗: 已启用")
	print("-" * 50)

	stats = convert_jsonl_to_bio(
		input_path=args.input,
		output_path=args.output,
		verbose=args.verbose,
		clean=args.clean
	)

	print_statistics(stats, clean=args.clean)


if __name__ == "__main__":
	main()
