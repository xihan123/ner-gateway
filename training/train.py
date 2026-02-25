#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NER 模型训练脚本
使用 BIO 格式的 JSONL 数据集微调 RoBERTa 中文模型进行命名实体识别
支持 CUDA 加速训练
"""

import json
import random
from pathlib import Path

import evaluate
import numpy as np
import torch
from datasets import Dataset
from transformers import (AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification,
                          EarlyStoppingCallback, Trainer, TrainingArguments)


def get_device():
	"""检测并返回最佳可用设备"""
	if torch.cuda.is_available():
		device = torch.device("cuda")
		print(f"  - 使用 CUDA: {torch.cuda.get_device_name(0)}")
		print(f"  - CUDA 版本: {torch.version.cuda}")
		print(f"  - 显存: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")
	elif torch.backends.mps.is_available():
		device = torch.device("mps")
		print("  - 使用 Apple MPS")
	else:
		device = torch.device("cpu")
		print("  - 使用 CPU")
	return device


# ===================== 配置参数 =====================
MODEL_NAME = "uer/roberta-medium-wwm-chinese-cluecorpussmall"
# 多数据源支持（按优先级顺序加载，自动合并）
DATA_PATHS = [
	"./bio_data.jsonl",  # AI 生成的客服场景数据
	"./bio_name_corpus.jsonl",  # 姓名语料库数据（中文+英文名）
]
OUTPUT_DIR = "./best_ner_model"

# 标签映射
LABEL_LIST = ["O", "B-PER", "I-PER"]
LABEL2ID = {label: i for i, label in enumerate(LABEL_LIST)}
ID2LABEL = {i: label for i, label in enumerate(LABEL_LIST)}

# 超参数（优化后）
BATCH_SIZE = 16                    # 降低 batch size，配合梯度累积
LEARNING_RATE = 2e-5               # 稍微降低学习率
NUM_EPOCHS = 15                    # 减少 epochs，配合早停
MAX_LENGTH = 128                   # 姓名识别任务 128 足够
GRADIENT_ACCUMULATION_STEPS = 2    # 梯度累积，有效 batch = 32
MAX_SAMPLES = 80000                # 最大样本数，避免数据过量


def load_jsonl_data(file_paths: list, max_samples: int = None) -> list:
	"""加载多个 JSONL 格式的数据文件并合并，支持采样"""
	data = []
	paths = file_paths if isinstance(file_paths, list) else [file_paths]

	for file_path in paths:
		if not Path(file_path).exists():
			print(f"  [跳过] 文件不存在: {file_path}")
			continue

		count = 0
		with open(file_path, "r", encoding="utf-8") as f:
			for line in f:
				if line.strip():
					data.append(json.loads(line))
					count += 1
		print(f"  [加载] {file_path}: {count} 条")

	# 如果数据量超过限制，随机采样
	if max_samples is not None and len(data) > max_samples:
		random.seed(42)
		data = random.sample(data, max_samples)
		print(f"  [采样] 数据量过大，随机采样至 {max_samples} 条")

	return data


def prepare_dataset(data: list) -> Dataset:
	"""将数据转换为 Dataset 格式，并将标签字符串转换为 ID"""
	processed_data = []
	for item in data:
		# 将标签字符串转换为数字 ID
		ner_tags = [LABEL2ID[label] for label in item["labels"]]
		processed_data.append({
			"tokens": item["tokens"],
			"ner_tags": ner_tags,
		})
	return Dataset.from_list(processed_data)


def tokenize_and_align_labels(examples, tokenizer):
	"""
	Tokenize 输入并对齐标签

	关键点：
	1. 中文字符通常会被分词器拆分为单个 token
	2. 需要处理特殊 token ([CLS], [SEP]) 的标签
	3. subword tokenization 会导致标签错位，需要正确对齐
	"""
	tokenized_inputs = tokenizer(
		examples["tokens"],
		truncation=True,
		is_split_into_words=True,
		max_length=MAX_LENGTH,
		padding=False,
	)

	labels = []
	for i, ner_tags in enumerate(examples["ner_tags"]):
		word_ids = tokenized_inputs.word_ids(batch_index=i)
		previous_word_idx = None
		label_ids = []

		for word_idx in word_ids:
			# 特殊 token (CLS, SEP 等) 设置为 -100，不参与损失计算
			if word_idx is None:
				label_ids.append(-100)
			# 只对每个词的第一个 token 标注标签，后续 subword 设为 -100
			elif word_idx != previous_word_idx:
				label_ids.append(ner_tags[word_idx])
			else:
				label_ids.append(-100)
			previous_word_idx = word_idx

		labels.append(label_ids)

	tokenized_inputs["labels"] = labels
	return tokenized_inputs


def compute_metrics(eval_pred):
	"""计算评估指标"""
	seqeval = evaluate.load("seqeval")
	predictions, labels = eval_pred
	predictions = np.argmax(predictions, axis=2)

	# 过滤掉 -100 (特殊 token 和 subword)
	true_predictions = [
		[LABEL_LIST[p] for (p, l) in zip(prediction, label) if l != -100]
		for prediction, label in zip(predictions, labels)
	]
	true_labels = [
		[LABEL_LIST[l] for (p, l) in zip(prediction, label) if l != -100]
		for prediction, label in zip(predictions, labels)
	]

	# 检查是否有有效预测
	if not true_predictions or all(len(p) == 0 for p in true_predictions):
		return {
			"precision": 0.0,
			"recall": 0.0,
			"f1": 0.0,
			"accuracy": 0.0,
		}

	results = seqeval.compute(predictions=true_predictions, references=true_labels, zero_division=0)
	return {
		"precision": results["overall_precision"],
		"recall": results["overall_recall"],
		"f1": results["overall_f1"],
		"accuracy": results["overall_accuracy"],
	}


def main():
	print("=" * 50)
	print("NER 模型训练")
	print("=" * 50)

	# 1. 加载数据
	print("\n[1/6] 加载数据...")
	raw_data = load_jsonl_data(DATA_PATHS, max_samples=MAX_SAMPLES)
	print(f"  - 总样本数: {len(raw_data)}")

	# 2. 转换为 Dataset
	print("\n[2/6] 准备数据集...")
	dataset = prepare_dataset(raw_data)

	# 划分训练集和验证集 (90% / 10%)
	split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
	train_dataset = split_dataset["train"]
	eval_dataset = split_dataset["test"]
	print(f"  - 训练集: {len(train_dataset)}")
	print(f"  - 验证集: {len(eval_dataset)}")

	# 3. 加载 Tokenizer 和模型
	print("\n[3/6] 加载模型和 Tokenizer...")
	tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
	model = AutoModelForTokenClassification.from_pretrained(
		MODEL_NAME,
		num_labels=len(LABEL_LIST),
		id2label=ID2LABEL,
		label2id=LABEL2ID,
		ignore_mismatched_sizes=True,
	)
	print(f"  - 模型: {MODEL_NAME}")
	print(f"  - 标签数: {len(LABEL_LIST)}")

	# 4. Tokenize 和对齐标签
	print("\n[4/6] Tokenize 数据...")
	tokenized_train = train_dataset.map(
		lambda x: tokenize_and_align_labels(x, tokenizer),
		batched=True,
		remove_columns=train_dataset.column_names,
	)
	tokenized_eval = eval_dataset.map(
		lambda x: tokenize_and_align_labels(x, tokenizer),
		batched=True,
		remove_columns=eval_dataset.column_names,
	)

	# 5. 配置训练参数
	print("\n[5/6] 配置训练参数...")

	# 检测设备
	device = get_device()
	use_cuda = device.type == "cuda"

	# 检测是否支持 bf16 (Ampere 及更新架构)
	use_bf16 = use_cuda and torch.cuda.is_bf16_supported()
	use_fp16 = use_cuda and not use_bf16

	if use_bf16:
		print("  - 启用 BF16 混合精度训练")
	elif use_fp16:
		print("  - 启用 FP16 混合精度训练")

	training_args = TrainingArguments(
		output_dir=OUTPUT_DIR,
		learning_rate=LEARNING_RATE,
		per_device_train_batch_size=BATCH_SIZE,
		per_device_eval_batch_size=BATCH_SIZE * 2,
		gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
		num_train_epochs=NUM_EPOCHS,
		max_grad_norm=1.0,
		weight_decay=0.01,
		eval_strategy="epoch",
		save_strategy="epoch",
		load_best_model_at_end=True,
		metric_for_best_model="f1",
		greater_is_better=True,
		logging_steps=20,
		warmup_ratio=0.1,
		report_to="none",
		seed=42,
		# 混合精度训练
		fp16=use_fp16,
		bf16=use_bf16,
		# 数据加载优化（Windows 兼容）
		dataloader_num_workers=0,
		dataloader_pin_memory=True,
		# 保存策略
		save_total_limit=3,
	)

	# 数据整理器
	data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

	# 初始化 Trainer
	trainer = Trainer(
		model=model,
		args=training_args,
		train_dataset=tokenized_train,
		eval_dataset=tokenized_eval,
		processing_class=tokenizer,
		data_collator=data_collator,
		compute_metrics=compute_metrics,
		callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
	)

	# 6. 开始训练
	print("\n[6/6] 开始训练...")
	print("-" * 50)
	trainer.train()

	# 保存最终模型
	print("\n" + "=" * 50)
	print("训练完成！保存模型...")
	output_path = Path(OUTPUT_DIR)
	output_path.mkdir(parents=True, exist_ok=True)

	# 保存模型和 tokenizer
	trainer.save_model(OUTPUT_DIR)
	tokenizer.save_pretrained(OUTPUT_DIR)

	print(f"模型已保存到: {OUTPUT_DIR}")
	print("=" * 50)

	# 最终评估
	print("\n最终评估结果:")
	eval_results = trainer.evaluate()
	for key, value in eval_results.items():
		if isinstance(value, float):
			print(f"  - {key}: {value:.4f}")


if __name__ == "__main__":
	main()
