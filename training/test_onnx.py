#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""测试 ONNX 模型（带可视化结果，支持长文本滑动窗口推理）"""
import re
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

# 标签映射
LABEL_LIST = ["O", "B-PER", "I-PER"]
ID2LABEL = {i: label for i, label in enumerate(LABEL_LIST)}

# 模型最大支持的 token 长度（BERT/RoBERTa 标准长度）
MAX_MODEL_LENGTH = 512


def clean_html(text: str) -> str:
	"""去除 HTML 标签"""
	text = re.sub(r"<[^>]+>", "", text)
	return text.strip()


def sliding_window_inference(session, tokenizer, text, chunk_size=480, overlap=32):
	"""
	滑动窗口推理，支持长文本
	
	Args:
		session: ONNX 推理会话
		tokenizer: 分词器
		text: 输入文本
		chunk_size: 每个窗口的 token 数量（不含 [CLS]/[SEP]）
		overlap: 窗口之间的重叠 token 数量
	
	Returns:
		predictions: 每个字符位置的预测标签 ID
	"""
	# 先对整个文本进行 tokenize 获取字符映射
	full_encoding = tokenizer(
		text,
		return_tensors="np",
		padding=False,
		truncation=False,
		return_offsets_mapping=True,
		add_special_tokens=False,
	)

	tokens = tokenizer.convert_ids_to_tokens(full_encoding["input_ids"][0])
	total_tokens = len(tokens)

	if total_tokens <= chunk_size:
		# 短文本直接推理
		encoding = tokenizer(
			text,
			return_tensors="np",
			padding="max_length",
			max_length=MAX_MODEL_LENGTH,
			truncation=True,
			return_offsets_mapping=True,
		)
		ort_inputs = {k: v.astype(np.int64) for k, v in encoding.items() if k != "offset_mapping"}
		logits = session.run(None, ort_inputs)[0]
		preds = np.argmax(logits, axis=-1)[0]

		# 提取有效 token 的预测（跳过 [CLS] 和 [SEP]）
		valid_len = min(total_tokens + 2, MAX_MODEL_LENGTH)
		return preds[1:valid_len - 1], full_encoding["offset_mapping"][0]

	# 长文本：滑动窗口推理
	# 初始化每个 token 的预测结果（存储所有窗口的预测）
	all_predictions = [[] for _ in range(total_tokens)]

	start = 0
	while start < total_tokens:
		end = min(start + chunk_size, total_tokens)

		# 获取当前窗口的 token 范围
		window_tokens = tokens[start:end]
		window_text = text[full_encoding["offset_mapping"][0][start][0]:full_encoding["offset_mapping"][0][end - 1][1]]

		# 对窗口文本进行 tokenize（带特殊 token）
		encoding = tokenizer(
			window_text,
			return_tensors="np",
			padding="max_length",
			max_length=MAX_MODEL_LENGTH,
			truncation=True,
			return_offsets_mapping=True,
		)

		ort_inputs = {k: v.astype(np.int64) for k, v in encoding.items() if k != "offset_mapping"}
		logits = session.run(None, ort_inputs)[0]
		preds = np.argmax(logits, axis=-1)[0]

		# 计算窗口有效长度
		window_token_count = min(end - start + 2, MAX_MODEL_LENGTH)  # +2 for [CLS] and [SEP]

		# 将窗口预测映射回原始 token 位置
		for i in range(window_token_count - 2):  # 跳过 [CLS] 和 [SEP]
			if start + i < total_tokens:
				all_predictions[start + i].append(preds[i + 1])  # +1 跳过 [CLS]

		# 滑动到下一个窗口
		start = end - overlap
		if start >= total_tokens - overlap:
			break

	# 对每个 token 位置取多数投票
	final_predictions = []
	for preds_list in all_predictions:
		if preds_list:
			# 多数投票，优先选择非 O 标签
			from collections import Counter
			vote = Counter(preds_list)
			# 如果有非 O 标签的预测，优先选择
			non_o_preds = [p for p in preds_list if p != 0]
			if non_o_preds:
				# 对于实体标签，需要更高的置信度（至少 2 次预测）
				non_o_counter = Counter(non_o_preds)
				most_common = non_o_counter.most_common(1)[0]
				# 如果预测次数 >= 2 或者只有一个预测结果，则接受
				if most_common[1] >= 2 or len(preds_list) == 1:
					final_predictions.append(most_common[0])
				else:
					# 单次预测可能不可靠，回退到多数投票
					final_predictions.append(vote.most_common(1)[0][0])
			else:
				final_predictions.append(vote.most_common(1)[0][0])
		else:
			final_predictions.append(0)  # 默认 O

	# 后处理：清理孤立的实体标签
	for i in range(len(final_predictions)):
		if final_predictions[i] == 2:  # I-PER
			# I-PER 前面必须是 B-PER 或 I-PER
			if i == 0 or final_predictions[i - 1] not in [1, 2]:
				final_predictions[i] = 0  # 改为 O
		elif final_predictions[i] == 1:  # B-PER
			# 检查是否是孤立的 B-PER（后面没有 I-PER）
			if i == len(final_predictions) - 1 or final_predictions[i + 1] not in [1, 2]:
				# 单字符人名可能是误判，检查上下文
				pass  # 保留，因为单字符人名也是可能的

	return np.array(final_predictions), full_encoding["offset_mapping"][0]


# 优先使用 FP16 模型（精度无损，体积减半）
fp16_path = "./onnx_models/model_fp16.onnx"
fp32_path = "./onnx_models/model.onnx"
int8_path = "./ner_model_int8.onnx"

if Path(int8_path).exists():
	model_path = int8_path
	print(f"使用 int8 模型: {model_path}")
elif Path(fp16_path).exists():
	model_path = fp16_path
	print(f"使用 FP16 模型: {model_path}")
elif Path(fp32_path).exists():
	model_path = fp32_path
	print(f"使用 FP32 模型: {model_path}")
else:
	model_path = "./ner_model.onnx"
	print(f"使用原始模型: {model_path}")

# 加载模型
session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
tokenizer = AutoTokenizer.from_pretrained("./best_ner_model")

# 测试文本（包含长文本）
raw_text = """
<div class=\"msg\">李明收到了华为的邮件，但张三丰满的衣服店一直没发货。</div><br>
""" #* 10  # 复制 10 次模拟长文本

# 预处理：去除 HTML 标签
text = clean_html(raw_text)

# 使用滑动窗口推理
predictions, offset_mapping = sliding_window_inference(session, tokenizer, text, chunk_size=450, overlap=64)
tokens = tokenizer.convert_ids_to_tokens(tokenizer(text, add_special_tokens=False)["input_ids"])

# ========== 可视化结果 ==========
print("=" * 60)
print(f"清洗文本长度: {len(text)} 字符, {len(tokens)} tokens")
print(f"文本预览: {text[:100]}...")
print("=" * 60)

# 只显示前 50 个 token 的预测结果
print("\n【Token 级别预测（前 50 个）】")
print("-" * 50)
for i, (token, pred_id) in enumerate(zip(tokens[:50], predictions[:50])):
	label = ID2LABEL[pred_id]
	# 获取原始文本片段
	start, end = offset_mapping[i]
	original = text[start:end] if end > start else token
	color = "\033[92m" if "PER" in label else ""
	reset = "\033[0m" if color else ""
	print(f"  {original:12} -> {color}{label}{reset}")

# ========== 实体抽取可视化 ==========
print("\n【抽取的命名实体】")
print("-" * 50)
entities = []
current_start = None
current_end = None
current_type = None

for i, pred_id in enumerate(predictions):
	if i >= len(offset_mapping):
		break
	label = ID2LABEL[pred_id]
	start, end = offset_mapping[i]

	if label.startswith("B-"):
		if current_type is not None:
			entities.append((text[current_start:current_end], current_type))
		current_start = start
		current_end = end
		current_type = label[2:]
	elif label.startswith("I-") and current_type == label[2:]:
		current_end = end
	else:
		if current_type is not None:
			entities.append((text[current_start:current_end], current_type))
			current_start = current_end = current_type = None

# 处理最后一个实体
if current_type is not None:
	entities.append((text[current_start:current_end], current_type))

# 去重并显示
seen = set()
if entities:
	for entity_text, entity_type in entities:
		if entity_text and entity_text not in seen:
			seen.add(entity_text)
			type_name = {"PER": "人名"}.get(entity_type, entity_type)
			print(f"  [{type_name}] \033[92m{entity_text}\033[0m")
else:
	print("  未检测到命名实体")

print("\n" + "=" * 60)
print(f"预测 tokens 数量: {len(predictions)}")
print("滑动窗口推理成功！")
