#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os
import random
import time
from typing import Any, Dict, List

from openai import OpenAI

COMMON_SURNAMES = [
	"王", "李", "张", "刘", "陈", "杨", "黄", "赵", "周", "吴",
	"徐", "孙", "马", "朱", "胡", "郭", "何", "高", "林", "罗",
	"郑", "梁", "谢", "宋", "唐", "许", "韩", "冯", "邓", "曹",
	"彭", "曾", "萧", "田", "董", "袁", "潘", "于", "蒋", "蔡",
	"余", "杜", "叶", "程", "苏", "魏", "吕", "丁", "任", "沈",
	"欧阳", "司马", "诸葛", "上官", "皇甫", "东方", "令狐", "慕容"
]

COMMON_GIVEN_NAMES = [
	"伟", "芳", "娜", "敏", "静", "丽", "强", "磊", "军", "洋",
	"勇", "艳", "杰", "涛", "明", "超", "秀", "霞", "平", "刚",
	"桂", "英", "华", "建", "文", "玲", "斌", "宇", "浩", "凯",
	"晨", "阳", "婷", "欣", "怡", "雪", "梅", "燕", "红", "云",
	"思", "雨", "佳", "慧", "琳", "博", "志", "国", "海", "成"
]

ENGLISH_NAMES = [
	"Tony", "Jack", "Mike", "David", "Tom", "Alex", "Kevin", "Eric",
	"Lucy", "Mary", "Lisa", "Emma", "Anna", "Grace", "Sophie", "Alice",
	"Andy", "Jerry", "Jason", "Peter", "Henry", "Frank", "Steven", "Tony"
]

SYSTEM_PROMPT = """中文姓名识别数据增强。BIO标签：O(非人名)、B-PER(人名起始)、I-PER(人名后续)。
只标真实人名，尊称如"张总"只标"张"，干扰项如"平安客服"不标注。
输出JSON：{"tokens": [...], "labels": [...]}"""

AUGMENT_PROMPT_TEMPLATE = """基于样本生成 {count} 条数据：

{samples}

要求：10-100字，40%改句式，30%换人名，20%新场景，10%干扰项。直接输出JSON数组。"""


def create_client() -> OpenAI:
	api_key = os.environ.get("OPENAI_API_KEY")
	if not api_key:
		raise ValueError("请设置环境变量 OPENAI_API_KEY")

	base_url = os.environ.get("OPENAI_BASE_URL")
	client_kwargs = {"api_key": api_key}
	if base_url:
		client_kwargs["base_url"] = base_url

	return OpenAI(**client_kwargs)


def load_bio_data(file_path: str) -> List[Dict[str, Any]]:
	data = []
	with open(file_path, "r", encoding="utf-8") as f:
		for line in f:
			line = line.strip()
			if line:
				try:
					item = json.loads(line)
					if "tokens" in item and "labels" in item:
						data.append(item)
				except json.JSONDecodeError:
					continue
	return data


def is_valid_name(name: str) -> bool:
	if not name or not name.strip():
		return False
	for ch in name:
		if not ch.isascii() and not ch.isspace():
			return True
		if ch.isascii() and ch.isalnum():
			return True
	return False


def extract_names_from_bio(item: Dict[str, Any]) -> List[str]:
	tokens = item["tokens"]
	labels = item["labels"]

	names = []
	current_name = []

	for token, label in zip(tokens, labels):
		if label == "B-PER":
			if current_name:
				name = "".join(current_name)
				if is_valid_name(name):
					names.append(name)
			current_name = [token]
		elif label == "I-PER" and current_name:
			current_name.append(token)
		else:
			if current_name:
				name = "".join(current_name)
				if is_valid_name(name):
					names.append(name)
				current_name = []

	if current_name:
		name = "".join(current_name)
		if is_valid_name(name):
			names.append(name)

	return names


def tokens_to_text(tokens: List[str]) -> str:
	return "".join(tokens)


def generate_random_name() -> str:
	if random.random() < 0.1:
		surname = random.choice([s for s in COMMON_SURNAMES if len(s) == 2])
	else:
		surname = random.choice([s for s in COMMON_SURNAMES if len(s) == 1])

	given_len = random.randint(1, 2)
	given = "".join(random.choices(COMMON_GIVEN_NAMES, k=given_len))

	return surname + given


def generate_augmented_batch(
		client: OpenAI,
		model: str,
		samples: List[Dict[str, Any]],
		count: int = 10,
		max_retries: int = 3
) -> List[Dict[str, Any]]:
	sample_texts = []
	for i, sample in enumerate(samples[:10]):
		text = tokens_to_text(sample["tokens"])
		names = extract_names_from_bio(sample)
		sample_texts.append(f"{i + 1}. 文本: {text}\n   人名: {', '.join(names) if names else '无'}")

	samples_str = "\n".join(sample_texts)
	user_prompt = AUGMENT_PROMPT_TEMPLATE.format(count=count, samples=samples_str)

	for attempt in range(max_retries):
		try:
			response = client.chat.completions.create(
				model=model,
				messages=[
					{"role": "system", "content": SYSTEM_PROMPT},
					{"role": "user", "content": user_prompt}
				],
				temperature=1.2,
				max_tokens=8000,
			)

			content = response.choices[0].message.content.strip()

			if content.startswith("```"):
				lines = content.split("\n")
				content = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

			data = json.loads(content)

			if isinstance(data, list):
				validated = []
				for item in data:
					if isinstance(item, dict) and "tokens" in item and "labels" in item:
						if len(item["tokens"]) == len(item["labels"]):
							validated.append(item)
				return validated

		except json.JSONDecodeError as e:
			print(f"[警告] JSON 解析失败 (尝试 {attempt + 1}/{max_retries}): {e}")
			if attempt < max_retries - 1:
				time.sleep(2)
		except Exception as e:
			print(f"[错误] API 调用失败 (尝试 {attempt + 1}/{max_retries}): {e}")
			if attempt < max_retries - 1:
				time.sleep(2)

	return []


def simple_augment(item: Dict[str, Any]) -> List[Dict[str, Any]]:
	results = []
	tokens = item["tokens"].copy()
	labels = item["labels"].copy()

	names = extract_names_from_bio(item)
	if not names:
		return results

	for _ in range(2):
		new_tokens = tokens.copy()
		new_labels = labels.copy()

		offset = 0
		for original_name in names:
			name_tokens = list(original_name)
			name_len = len(name_tokens)

			for i in range(len(tokens) - name_len + 1):
				if tokens[i:i + name_len] == name_tokens:
					if labels[i] == "B-PER":
						new_name = generate_random_name()
						new_name_tokens = list(new_name)

						actual_i = i + offset
						new_tokens = new_tokens[:actual_i] + new_name_tokens + new_tokens[actual_i + name_len:]

						new_labels_part = ["B-PER"] + ["I-PER"] * (len(new_name_tokens) - 1)
						new_labels = new_labels[:actual_i] + new_labels_part + new_labels[actual_i + name_len:]

						offset += len(new_name_tokens) - name_len
						break

		results.append({"tokens": new_tokens, "labels": new_labels})

	return results


def merge_bio_files(input_paths: List[str]) -> List[Dict[str, Any]]:
	all_data = []
	for path in input_paths:
		if os.path.exists(path):
			data = load_bio_data(path)
			print(f"[信息] 加载 {path}: {len(data)} 条")
			all_data.extend(data)
	return all_data


def filter_quality_samples(data: List[Dict[str, Any]], max_length: int = 512) -> List[Dict[str, Any]]:
	filtered = []
	for item in data:
		tokens = item.get("tokens", [])
		labels = item.get("labels", [])

		if len(tokens) > max_length or len(tokens) < 5:
			continue

		if len(tokens) != len(labels):
			continue

		if all(l == "O" for l in labels) and random.random() > 0.2:
			continue

		filtered.append(item)

	return filtered


def main():
	parser = argparse.ArgumentParser(description="NER数据增强器")

	parser.add_argument(
		"--input", "-i",
		type=str,
		nargs="+",
		required=True,
		help="输入 BIO 格式文件路径（支持多个文件）"
	)

	parser.add_argument(
		"--output", "-o",
		type=str,
		required=True,
		help="输出文件路径"
	)

	parser.add_argument(
		"--count", "-c",
		type=int,
		default=500,
		help="生成数据条数 (默认: 500)"
	)

	parser.add_argument(
		"--batch-size", "-b",
		type=int,
		default=20,
		help="每批生成数量 (默认: 20)"
	)

	parser.add_argument(
		"--model", "-m",
		type=str,
		default=None,
		help="模型名称 (默认: 从环境变量读取或 deepseek-chat)"
	)

	parser.add_argument(
		"--simple-only",
		action="store_true",
		help="只使用简单增强（不调用 API）"
	)

	parser.add_argument(
		"--append", "-a",
		action="store_true",
		help="追加到现有文件"
	)

	parser.add_argument(
		"--include-original",
		action="store_true",
		help="输出中包含原始数据"
	)

	args = parser.parse_args()

	print(f"[信息] 加载输入数据...")
	original_data = merge_bio_files(args.input)
	print(f"[信息] 原始数据总量: {len(original_data)} 条")

	original_data = filter_quality_samples(original_data)
	print(f"[信息] 过滤后数据量: {len(original_data)} 条")

	if not original_data:
		print("[错误] 没有有效的输入数据")
		return

	with_names = sum(1 for item in original_data if any(l != "O" for l in item["labels"]))
	print(f"[信息] 包含人名的样本: {with_names} 条 ({with_names / len(original_data) * 100:.1f}%)")

	augmented_data = []

	print(f"\n[进度] 执行简单增强...")
	for item in original_data:
		simple_results = simple_augment(item)
		augmented_data.extend(simple_results)

	print(f"[信息] 简单增强生成: {len(augmented_data)} 条")

	if not args.simple_only:
		print(f"\n[进度] 执行 API 增强...")

		client = create_client()
		model = args.model or os.environ.get("OPENAI_MODEL", "deepseek-chat")

		print(f"[信息] 使用模型: {model}")

		api_generated = 0
		batches_needed = (args.count - len(augmented_data) + args.batch_size - 1) // args.batch_size
		batches_needed = max(0, batches_needed)

		reference_samples = random.sample(original_data, min(20, len(original_data)))

		for batch_idx in range(batches_needed):
			remaining = args.count - len(augmented_data) - api_generated
			if remaining <= 0:
				break

			current_batch_size = min(args.batch_size, remaining)

			print(f"[进度] API 生成第 {batch_idx + 1}/{batches_needed} 批...")

			batch_data = generate_augmented_batch(
				client, model, reference_samples, current_batch_size
			)

			if batch_data:
				batch_data = filter_quality_samples(batch_data)
				augmented_data.extend(batch_data)
				api_generated += len(batch_data)
				print(f"[成功] 本批生成 {len(batch_data)} 条，累计 {len(augmented_data)} 条")
			else:
				print(f"[失败] 本批生成失败，跳过")

			if batch_idx < batches_needed - 1:
				time.sleep(1)

		print(f"[信息] API 增强生成: {api_generated} 条")

	if len(augmented_data) > args.count:
		augmented_data = random.sample(augmented_data, args.count)

	if args.include_original:
		augmented_data = original_data + augmented_data
		print(f"[信息] 包含原始数据后总量: {len(augmented_data)} 条")

	write_mode = "a" if args.append else "w"
	os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

	with open(args.output, write_mode, encoding="utf-8") as f:
		for item in augmented_data:
			f.write(json.dumps(item, ensure_ascii=False) + "\n")

	print(f"\n[完成] 生成 {len(augmented_data)} 条数据，已保存至 {args.output}")


if __name__ == "__main__":
	main()
