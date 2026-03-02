import argparse
import json
import random
import time
import traceback
from pathlib import Path

import evaluate
import numpy as np
import onnxruntime as ort
import torch
from optimum.onnxruntime import ORTModelForTokenClassification
from transformers import AutoModelForTokenClassification, AutoTokenizer

LABELS = ["O", "B-PER", "I-PER"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}


def load_bio_dataset(path: str, max_samples: int = None) -> list:
	data = []
	with open(path, "r", encoding="utf-8") as f:
		for line in f:
			if not line.strip():
				continue
			item = json.loads(line)
			if "tokens" in item and "labels" in item:
				data.append({
					"tokens": item["tokens"],
					"ner_tags": [LABEL2ID.get(l, 0) for l in item["labels"]],
				})
	if max_samples and len(data) > max_samples:
		random.seed(42)
		data = random.sample(data, max_samples)
	return data


def evaluate_model(model_dir: str, test_data: list, is_onnx: bool, batch_size: int = 32) -> dict:
	tokenizer = AutoTokenizer.from_pretrained(model_dir)

	if is_onnx:
		model, provider = _load_onnx_model(model_dir)
	else:
		model, provider = _load_pytorch_model(model_dir)

	print(f"  设备: {provider} | Batch: {batch_size}")

	seqeval = evaluate.load("seqeval")
	predictions, references = [], []
	total_time = 0.0

	for i in range(0, len(test_data), batch_size):
		batch = test_data[i:i + batch_size]
		tokens = [s["tokens"] for s in batch]
		true_labels = [s["ner_tags"] for s in batch]

		inputs = tokenizer(
			tokens, return_tensors="pt" if not is_onnx else "np",
			is_split_into_words=True, truncation=True, max_length=512, padding=True
		)

		if not is_onnx and torch.cuda.is_available():
			inputs = {k: v.to("cuda") for k, v in inputs.items()}

		t0 = time.perf_counter()
		if is_onnx:
			logits = model(**inputs).logits
		else:
			with torch.no_grad():
				logits = model(**inputs).logits.cpu().numpy()
		total_time += time.perf_counter() - t0

		pred_ids = np.argmax(logits, axis=-1)

		for b_idx in range(len(batch)):
			word_ids = inputs.word_ids(batch_index=b_idx)
			pred = []
			prev = None
			for idx, wid in enumerate(word_ids):
				if wid is not None and wid != prev:
					pred.append(LABELS[pred_ids[b_idx][idx]])
				prev = wid

			min_len = min(len(pred), len(true_labels[b_idx]))
			predictions.append(pred[:min_len])
			references.append([LABELS[l] for l in true_labels[b_idx][:min_len]])

	results = seqeval.compute(predictions=predictions, references=references, zero_division=0)
	return {
		"f1": results["overall_f1"],
		"precision": results["overall_precision"],
		"recall": results["overall_recall"],
		"num_samples": len(test_data),
		"avg_time_ms": total_time / len(test_data) * 1000,
		"throughput": len(test_data) / total_time,
		"provider": provider,
	}


def _load_onnx_model(model_dir: str):
	sess_opts = ort.SessionOptions()
	sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

	model_path = Path(model_dir)
	onnx_files = list(model_path.glob("*.onnx"))
	file_name = onnx_files[0].name if onnx_files else None

	if torch.cuda.is_available():
		try:
			model = ORTModelForTokenClassification.from_pretrained(
				model_dir, provider="CUDAExecutionProvider",
				file_name=file_name, session_options=sess_opts
			)
			model.use_io_binding = False
			return model, model.providers[0]
		except Exception:
			pass

	model = ORTModelForTokenClassification.from_pretrained(
		model_dir, provider="CPUExecutionProvider",
		file_name=file_name, session_options=sess_opts
	)
	return model, "CPU"


def _load_pytorch_model(model_dir: str):
	model = AutoModelForTokenClassification.from_pretrained(model_dir)
	device = "cuda" if torch.cuda.is_available() else "cpu"
	model.to(device).eval()
	return model, device


def main():
	parser = argparse.ArgumentParser(description="NER 模型基准测试")
	parser.add_argument("-m", "--model", action="append", required=True, help="模型目录")
	parser.add_argument("-t", "--test-data", default="raw_data/test_bio_name_corpus.jsonl")
	parser.add_argument("--max-samples", type=int, default=50000)
	parser.add_argument("-b", "--batch-size", type=int, default=32)
	parser.add_argument("--onnx", action="store_true", help="模型为 ONNX 格式")
	parser.add_argument("-o", "--output", default="benchmark_report.md")
	args = parser.parse_args()

	test_data = load_bio_dataset(args.test_data, args.max_samples)
	print(f"测试集: {len(test_data)} 条")

	results = {}
	for path in args.model:
		is_onnx = args.onnx or "onnx" in path.lower()
		if not Path(path).exists():
			print(f"跳过: {path} (不存在)")
			continue
		try:
			print(f"\n{'=' * 50}\n模型: {path}")
			r = evaluate_model(path, test_data, is_onnx, args.batch_size)
			results[path] = r
			print(f"  F1: {r['f1']:.4f} | P: {r['precision']:.4f} | R: {r['recall']:.4f}")
			print(f"  时延: {r['avg_time_ms']:.2f}ms | 吞吐: {r['throughput']:.1f}/s")
		except Exception as e:
			print(f"错误: {e}")
			traceback.print_exc()

	# 生成报告
	lines = ["# NER 模型性能对比", "", "| 模型 | F1 | Precision | Recall | 时延 | 吞吐 |", "|---|---|---|---|---|---|"]
	for p, r in sorted(results.items(), key=lambda x: x[1]["f1"], reverse=True):
		name = Path(p).name
		lines.append(
			f"| {name} | {r['f1']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} | {r['avg_time_ms']:.2f} | {r['throughput']:.1f} |")
	Path(args.output).write_text("\n".join(lines), encoding="utf-8")
	print(f"\n报告: {args.output}")


if __name__ == "__main__":
	main()