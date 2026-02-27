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
MAX_LENGTH = 512

# 标签映射（与训练脚本一致）
LABEL_LIST = ["O", "B-PER", "I-PER"]


def resolve_path(relative_path: str) -> Path:
	"""将相对于脚本目录的路径解析为绝对路径"""
	script_dir = Path(__file__).parent
	return (script_dir / relative_path).resolve()


def validate_model_dir(model_dir: str) -> None:
	"""
	验证模型目录是否包含有效的模型文件
	检查 model.safetensors 或 pytorch_model.bin 是否存在且非空
	"""
	model_path = resolve_path(model_dir)
	
	if not model_path.exists():
		raise FileNotFoundError(f"模型目录不存在: {model_path}")
	
	# 检查必要的配置文件
	config_file = model_path / "config.json"
	if not config_file.exists():
		raise FileNotFoundError(f"配置文件不存在: {config_file}")
	if config_file.stat().st_size == 0:
		raise ValueError(f"配置文件为空: {config_file}")
	
	# 检查模型权重文件
	safetensors_file = model_path / "model.safetensors"
	pytorch_model_file = model_path / "pytorch_model.bin"
	
	if safetensors_file.exists():
		file_size = safetensors_file.stat().st_size
		if file_size == 0:
			raise ValueError(
				f"模型文件为空 (0 字节): {safetensors_file}\n"
				f"请重新训练模型: uv run python training/train.py"
			)
		print(f"  - 模型目录: {model_path}")
		print(f"  - 找到模型文件: model.safetensors ({file_size / 1024 / 1024:.2f} MB)")
	elif pytorch_model_file.exists():
		file_size = pytorch_model_file.stat().st_size
		if file_size == 0:
			raise ValueError(
				f"模型文件为空 (0 字节): {pytorch_model_file}\n"
				f"请重新训练模型: uv run python training/train.py"
			)
		print(f"  - 模型目录: {model_path}")
		print(f"  - 找到模型文件: pytorch_model.bin ({file_size / 1024 / 1024:.2f} MB)")
	else:
		raise FileNotFoundError(
			f"未找到模型权重文件 (model.safetensors 或 pytorch_model.bin)\n"
			f"模型目录: {model_path}\n"
			f"请先训练模型: uv run python training/train.py"
		)


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

	# 解析绝对路径
	model_dir = resolve_path(MODEL_DIR)
	onnx_output = resolve_path(ONNX_OUTPUT)
	onnx_int8_output = resolve_path(ONNX_INT8_OUTPUT)
	vocab_output = resolve_path(VOCAB_OUTPUT)

	# 1. 验证模型文件
	print("\n[1/6] 验证模型文件...")
	validate_model_dir(MODEL_DIR)

	# 2. 加载模型和 Tokenizer
	print("\n[2/6] 加载模型...")
	tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
	model = AutoModelForTokenClassification.from_pretrained(str(model_dir))
	model.eval()

	print(f"  - 模型路径: {model_dir}")
	print(f"  - 标签数: {model.config.num_labels}")

	# 3. 构造 dummy input 并导出 ONNX
	print("\n[3/6] 导出 ONNX 模型...")

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
		str(onnx_output),
		input_names=input_names,
		output_names=output_names,
		dynamic_axes=dynamic_axes,
		opset_version=14,
		do_constant_folding=True,
		dynamo=False,  # 使用传统导出方式
	)
	print(f"  - ONNX 模型已保存: {onnx_output}")

	# 4. INT8 动态量化
	print("\n[4/6] INT8 动态量化...")
	from onnxruntime.quantization import quantize_dynamic, QuantType

	quantize_dynamic(
		model_input=str(onnx_output),
		model_output=str(onnx_int8_output),
		weight_type=QuantType.QInt8,  # 使用 QInt8 进行权重量化
	)
	print(f"  - INT8 量化模型已保存: {onnx_int8_output}")

	# 5. 导出 vocab.txt
	print("\n[5/6] 导出词汇表...")
	export_vocab(tokenizer, str(vocab_output))

	# 6. 验证量化模型
	print("\n[6/6] 验证量化模型...")
	test_onnx_model(str(onnx_int8_output), tokenizer)

	# 打印模型大小对比
	print("\n" + "=" * 50)
	print("输出文件:")
	original_size = onnx_output.stat().st_size / 1024 / 1024
	quantized_size = onnx_int8_output.stat().st_size / 1024 / 1024
	vocab_size_kb = vocab_output.stat().st_size / 1024
	print(f"  - ONNX 模型: {onnx_output} ({original_size:.2f} MB)")
	print(f"  - INT8 量化: {onnx_int8_output} ({quantized_size:.2f} MB)")
	print(f"  - 词汇表: {vocab_output} ({vocab_size_kb:.1f} KB)")
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
