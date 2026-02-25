#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ONNX 导出与量化脚本
将微调后的 NER 模型导出为 ONNX 格式并进行 INT8 动态量化
"""

from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

# ===================== 配置 =====================
MODEL_DIR = "../models/best_ner_model"
ONNX_OUTPUT = "../models/ner_model.onnx"
ONNX_INT8_OUTPUT = "../models/ner_model_int8.onnx"
VOCAB_OUTPUT = "../models/vocab.txt"
MAX_LENGTH = 128

# 标签映射（与训练脚本一致）
LABEL_LIST = ["O", "B-PER", "I-PER"]


def export_vocab(tokenizer, output_path: str):
	"""
	导出词汇表为 vocab.txt 格式
	格式：每行一个 token，行号即为对应的 token ID
	"""
	vocab = tokenizer.get_vocab()
	# 按ID排序
	sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])

	with open(output_path, "w", encoding="utf-8") as f:
		for token, token_id in sorted_vocab:
			f.write(f"{token}\n")

	print(f"  - 词汇表已保存: {output_path}")
	print(f"  - 词汇表大小: {len(sorted_vocab)}")


def export_onnx():
	"""导出 PyTorch 模型为 ONNX 格式"""
	print("=" * 50)
	print("ONNX 导出与量化")
	print("=" * 50)

	# 1. 加载模型和 Tokenizer
	print("\n[1/4] 加载模型...")
	tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
	model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)
	model.eval()

	print(f"  - 模型路径: {MODEL_DIR}")
	print(f"  - 标签数: {model.config.num_labels}")

	# 2. 构造 dummy input
	print("\n[2/4] 导出 ONNX 模型...")

	# 创建示例输入（支持动态 batch_size 和 sequence_length）
	dummy_text = "这是一个测试句子"
	dummy_inputs = tokenizer(
		dummy_text,
		return_tensors="pt",
		padding="max_length",
		truncation=True,
		max_length=MAX_LENGTH,
	)

	# 输入名称
	input_names = ["input_ids", "attention_mask", "token_type_ids"]
	output_names = ["logits"]

	# 动态轴配置：支持不同的 batch_size 和 sequence_length
	dynamic_axes = {
		"input_ids": {0: "batch_size", 1: "sequence_length"},
		"attention_mask": {0: "batch_size", 1: "sequence_length"},
		"token_type_ids": {0: "batch_size", 1: "sequence_length"},
		"logits": {0: "batch_size", 1: "sequence_length"},
	}

	# 导出 ONNX（使用传统导出方式，确保权重完整导出）
	torch.onnx.export(
		model,
		(dummy_inputs["input_ids"], dummy_inputs["attention_mask"], dummy_inputs["token_type_ids"]),
		ONNX_OUTPUT,
		input_names=input_names,
		output_names=output_names,
		dynamic_axes=dynamic_axes,
		opset_version=14,
		do_constant_folding=True,
		dynamo=False,  # 使用传统导出方式
	)
	print(f"  - ONNX 模型已保存: {ONNX_OUTPUT}")

	# 3. INT8 动态量化
	print("\n[3/4] INT8 动态量化...")
	from onnxruntime.quantization import quantize_dynamic, QuantType

	quantize_dynamic(
		model_input=ONNX_OUTPUT,
		model_output=ONNX_INT8_OUTPUT,
		weight_type=QuantType.QInt8,  # 使用 QInt8 进行权重量化
	)
	print(f"  - INT8 量化模型已保存: {ONNX_INT8_OUTPUT}")

	# 4. 导出 vocab.txt
	print("\n[4/5] 导出词汇表...")
	export_vocab(tokenizer, VOCAB_OUTPUT)

	# 5. 验证量化模型
	print("\n[5/5] 验证量化模型...")
	test_onnx_model(ONNX_INT8_OUTPUT, tokenizer)

	# 打印模型大小对比
	print("\n" + "=" * 50)
	print("输出文件:")
	original_size = Path(ONNX_OUTPUT).stat().st_size / 1024 / 1024
	quantized_size = Path(ONNX_INT8_OUTPUT).stat().st_size / 1024 / 1024
	vocab_size = Path(VOCAB_OUTPUT).stat().st_size / 1024
	print(f"  - ONNX 模型: {ONNX_OUTPUT} ({original_size:.2f} MB)")
	print(f"  - INT8 量化: {ONNX_INT8_OUTPUT} ({quantized_size:.2f} MB)")
	print(f"  - 词汇表: {VOCAB_OUTPUT} ({vocab_size:.1f} KB)")
	print(f"  - 压缩比例: {original_size / quantized_size:.2f}x")
	print("=" * 50)


def test_onnx_model(onnx_path: str, tokenizer):
	"""测试 ONNX 模型推理"""
	# 创建 ONNX Runtime 会话
	sess_options = ort.SessionOptions()
	sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

	session = ort.InferenceSession(
		onnx_path,
		sess_options,
		providers=["CPUExecutionProvider"],
	)

	# 测试句子
	test_text = "张三在北京工作"
	print(f"  - 测试句子: {test_text}")

	# Tokenize
	inputs = tokenizer(
		test_text,
		return_tensors="np",
		padding="max_length",
		truncation=True,
		max_length=MAX_LENGTH,
	)

	# ONNX 推理
	ort_inputs = {
		"input_ids": inputs["input_ids"].astype(np.int64),
		"attention_mask": inputs["attention_mask"].astype(np.int64),
		"token_type_ids": inputs["token_type_ids"].astype(np.int64),
	}

	outputs = session.run(None, ort_inputs)
	logits = outputs[0]  # shape: [1, seq_len, num_labels]

	# 获取预测标签
	predictions = np.argmax(logits, axis=-1)[0]

	# 解码 tokens 并显示预测结果
	tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
	print(f"\n  - 预测结果:")
	for token, pred_id in zip(tokens[:len(test_text) + 2], predictions[:len(test_text) + 2]):
		if token not in ["[PAD]", "[CLS]", "[SEP]"]:
			label = LABEL_LIST[pred_id]
			print(f"    {token}: {label}")

	print(f"\n  - Logits shape: {logits.shape}")
	print(f"  - 模型转换成功！")


if __name__ == "__main__":
	export_onnx()
