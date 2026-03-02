#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import random
import re
from pathlib import Path

import evaluate
import numpy as np
import torch
from datasets import Dataset
from transformers import (
	AutoModelForTokenClassification,
	AutoTokenizer,
	DataCollatorForTokenClassification,
	EarlyStoppingCallback,
	Trainer,
	TrainingArguments
)

MODEL_NAME = "uer/roberta-medium-wwm-chinese-cluecorpussmall"
OUTPUT_DIR = "../models/best_ner_model"

PROCESSED_DATA_DIR = Path(__file__).parent.parent / "processed_data"
PROCESSED_TRAIN = PROCESSED_DATA_DIR / "train.jsonl"
PROCESSED_VAL = PROCESSED_DATA_DIR / "val.jsonl"
PROCESSED_TEST = PROCESSED_DATA_DIR / "test.jsonl"

RAW_DATA_DIR = Path(__file__).parent.parent / "raw_data"
RAW_DATA_PATHS = [RAW_DATA_DIR / "bio_name_corpus.jsonl"]
RAW_TRAIN_ONLY_PATHS = [
	RAW_DATA_DIR / "negative_samples.jsonl",
	RAW_DATA_DIR / "training_data.jsonl",
]

LABEL_LIST = ["O", "B-PER", "I-PER"]
LABEL2ID = {label: i for i, label in enumerate(LABEL_LIST)}
ID2LABEL = {i: label for i, label in enumerate(LABEL_LIST)}

BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 15
MAX_LENGTH = 128
GRADIENT_ACCUMULATION_STEPS = 2

# 预编译正则
RE_URL = re.compile(r'https?://|www\.|<[^>]+>', re.IGNORECASE)
RE_PUNCT = re.compile(
	r'^[\s\u3000-\u303F\uFF00-\uFFEF\u2000-\u206F\u00A0-\u00BF!"#$%&\'()*+,\-./:;<=>?@\[\\\]^_`{|}~]+$')

# 全局加载评估器
SEQEVAL = None


def get_device():
	if torch.cuda.is_available():
		print(f"使用 CUDA: {torch.cuda.get_device_name(0)}")
		return torch.device("cuda")
	elif torch.backends.mps.is_available():
		print("使用 Apple MPS")
		return torch.device("mps")
	print("使用 CPU")
	return torch.device("cpu")


def validate_bio_item(item: dict) -> bool:
	tokens, labels = item.get("tokens", []), item.get("labels", [])
	if not tokens or not labels or len(tokens) != len(labels):
		return False
	if any(l not in {"O", "B-PER", "I-PER"} for l in labels):
		return False
	if RE_URL.search("".join(tokens)):
		return False
	clean_tokens = [t for t in tokens if t.strip() and not RE_PUNCT.match(t)]
	return len(clean_tokens) >= 2


def clean_bio_data(data: list) -> list:
	cleaned = [item for item in data if validate_bio_item(item) and len(item["tokens"]) >= 3]
	print(f"清洗: {len(data)} -> {len(cleaned)} 条")
	return cleaned


def load_jsonl_file(file_path: Path) -> list:
	if not file_path.exists():
		return []
	data = []
	with open(file_path, "r", encoding="utf-8") as f:
		for line in f:
			line = line.strip()
			if line:
				data.append(json.loads(line))
	return data


def load_processed_datasets() -> tuple:
	if not all(p.exists() for p in [PROCESSED_TRAIN, PROCESSED_VAL]):
		return None

	print("\n使用 processed_data/ 数据集")
	train_data = load_jsonl_file(PROCESSED_TRAIN)
	val_data = load_jsonl_file(PROCESSED_VAL)
	test_data = load_jsonl_file(PROCESSED_TEST) if PROCESSED_TEST.exists() else []

	print(f"训练集: {len(train_data)}, 验证集: {len(val_data)}", end="")
	if test_data:
		print(f", 测试集: {len(test_data)}")
	else:
		print()

	return train_data, val_data, test_data


def load_raw_datasets() -> tuple:
	print("\n使用 raw_data/ 原始数据")

	raw_data = []
	for file_path in RAW_DATA_PATHS:
		raw_data.extend(load_jsonl_file(file_path))

	train_only_data = []
	for file_path in RAW_TRAIN_ONLY_PATHS:
		train_only_data.extend(load_jsonl_file(file_path))

	if raw_data:
		raw_data = clean_bio_data(raw_data)
	if train_only_data:
		train_only_data = clean_bio_data(train_only_data)

	if raw_data:
		random.seed(42)
		random.shuffle(raw_data)
		split_idx = int(len(raw_data) * 0.9)
		train_data = raw_data[:split_idx]
		eval_data = raw_data[split_idx:]
	else:
		train_data = []
		eval_data = []

	if train_only_data:
		train_data = train_data + train_only_data

	print(f"训练集: {len(train_data)}, 验证集: {len(eval_data)}")
	return train_data, eval_data, []


def prepare_dataset(data: list) -> Dataset:
	processed = [
		{"tokens": item["tokens"], "ner_tags": [LABEL2ID[l] for l in item["labels"]]}
		for item in data
	]
	return Dataset.from_list(processed)


def tokenize_and_align_labels(examples, tokenizer):
	tokenized = tokenizer(
		examples["tokens"],
		truncation=True,
		is_split_into_words=True,
		max_length=MAX_LENGTH,
		padding=False
	)

	labels = []
	for i, ner_tags in enumerate(examples["ner_tags"]):
		word_ids = tokenized.word_ids(batch_index=i)
		prev_idx = None
		label_ids = []
		for word_idx in word_ids:
			if word_idx is None:
				label_ids.append(-100)
			elif word_idx != prev_idx:
				label_ids.append(ner_tags[word_idx])
			else:
				label_ids.append(-100)
			prev_idx = word_idx
		labels.append(label_ids)

	tokenized["labels"] = labels
	return tokenized


def compute_metrics(eval_pred):
	global SEQEVAL
	if SEQEVAL is None:
		SEQEVAL = evaluate.load("seqeval")

	predictions, labels = eval_pred
	predictions = np.argmax(predictions, axis=2)

	true_predictions = [[LABEL_LIST[p] for (p, l) in zip(pred, label) if l != -100]
	                    for pred, label in zip(predictions, labels)]
	true_labels = [[LABEL_LIST[l] for (p, l) in zip(pred, label) if l != -100]
	               for pred, label in zip(predictions, labels)]

	results = SEQEVAL.compute(predictions=true_predictions, references=true_labels, zero_division=0)
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

	datasets = load_processed_datasets()
	if datasets is None:
		print("processed_data/ 不存在，从 raw_data/ 加载")
		datasets = load_raw_datasets()

	train_data, eval_data, test_data = datasets

	if not train_data:
		print("错误: 没有训练数据")
		return

	train_dataset = prepare_dataset(train_data)
	eval_dataset = prepare_dataset(eval_data) if eval_data else train_dataset

	tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
	model = AutoModelForTokenClassification.from_pretrained(
		MODEL_NAME, num_labels=len(LABEL_LIST), id2label=ID2LABEL, label2id=LABEL2ID, ignore_mismatched_sizes=True
	)

	tokenized_train = train_dataset.map(
		lambda x: tokenize_and_align_labels(x, tokenizer),
		batched=True,
		remove_columns=train_dataset.column_names
	)
	tokenized_eval = eval_dataset.map(
		lambda x: tokenize_and_align_labels(x, tokenizer),
		batched=True,
		remove_columns=eval_dataset.column_names
	)

	device = get_device()
	use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
	use_fp16 = device.type == "cuda" and not use_bf16

	training_args = TrainingArguments(
		output_dir=OUTPUT_DIR,
		learning_rate=LEARNING_RATE,
		per_device_train_batch_size=BATCH_SIZE,
		per_device_eval_batch_size=BATCH_SIZE * 2,
		gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
		num_train_epochs=NUM_EPOCHS,
		eval_strategy="epoch",
		save_strategy="epoch",
		load_best_model_at_end=True,
		metric_for_best_model="f1",
		greater_is_better=True,
		fp16=use_fp16,
		bf16=use_bf16,
		save_total_limit=3,
		dataloader_pin_memory=True,
		report_to="none"
	)

	trainer = Trainer(
		model=model,
		args=training_args,
		train_dataset=tokenized_train,
		eval_dataset=tokenized_eval,
		processing_class=tokenizer,
		data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer),
		compute_metrics=compute_metrics,
		callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
	)

	print("\n开始训练...")
	trainer.train()

	Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
	trainer.save_model(OUTPUT_DIR)
	tokenizer.save_pretrained(OUTPUT_DIR)
	print(f"\n模型已保存: {OUTPUT_DIR}")

	if test_data:
		print("\n测试集评估...")
		test_dataset = prepare_dataset(test_data)
		tokenized_test = test_dataset.map(
			lambda x: tokenize_and_align_labels(x, tokenizer),
			batched=True,
			remove_columns=test_dataset.column_names
		)
		test_results = trainer.evaluate(tokenized_test)
		print(f"F1: {test_results['eval_f1']:.4f}")
		print(f"Precision: {test_results['eval_precision']:.4f}")
		print(f"Recall: {test_results['eval_recall']:.4f}")


if __name__ == "__main__":
	main()