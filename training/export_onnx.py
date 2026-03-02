#!/usr/bin/env python
import argparse
import shutil
from pathlib import Path

import onnx
from datasets import Dataset
from onnxruntime.transformers.float16 import convert_float_to_float16
from optimum.onnxruntime import ORTModelForTokenClassification, ORTQuantizer
from optimum.onnxruntime.configuration import (
    AutoQuantizationConfig,
    CalibrationConfig,
    CalibrationMethod,
)
from transformers import AutoTokenizer, pipeline

MODEL_DIR = "../models/best_ner_model"
ONNX_OUTPUT_DIR = "../models/onnx_ner_model"
ONNX_INT8_OUTPUT_DIR = "../models/onnx_ner_model_int8"
ONNX_FP16_OUTPUT_DIR = "../models/onnx_ner_model_fp16"
VOCAB_OUTPUT = "../models/vocab.txt"

CALIBRATION_SAMPLES = [
    "张三今天在北京开会，见到了李四。",
    "王五打电话给赵六，说明天要去上海。",
    "请问您是陈先生吗？我是客服小李。",
    "您好，我是刘经理，这是我的名片。",
    "欧阳明和王小明一起去了公司。",
]


def resolve_path(relative_path: str) -> Path:
    return (Path(__file__).parent / relative_path).resolve()


def export_vocab(tokenizer, output_path: str):
    vocab = sorted(tokenizer.get_vocab().items(), key=lambda x: x[1])
    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(f"{token}\n" for token, _ in vocab)


def export_static_quantized(quantizer, tokenizer, onnx_int8_output):
    def preprocess(examples):
        return tokenizer(examples["text"], max_length=128, padding="max_length", truncation=True)

    calibration_dataset = (
        Dataset.from_dict({"text": CALIBRATION_SAMPLES})
        .map(preprocess, batched=True)
        .remove_columns(["text"])
    )
    calibration_dataset.set_format(type="np")

    calibration_config = CalibrationConfig(
        dataset_name="custom",
        dataset_config_name="default",
        dataset_split="train",
        dataset_num_samples=len(CALIBRATION_SAMPLES),
        method=CalibrationMethod.MinMax,
    )

    calibration_tensors_range = quantizer.fit(
        dataset=calibration_dataset,
        calibration_config=calibration_config,
    )

    qconfig = AutoQuantizationConfig.arm64(is_static=True, per_channel=False)
    quantizer.quantize(
        save_dir=str(onnx_int8_output),
        quantization_config=qconfig,
        calibration_tensors_range=calibration_tensors_range,
    )
    tokenizer.save_pretrained(str(onnx_int8_output))
    print(f"INT8 静态量化: {onnx_int8_output}")


def export_fp16(ort_model, tokenizer, onnx_fp16_output):
    model_path = ort_model.model_path
    onnx_fp16_output.mkdir(parents=True, exist_ok=True)

    model = onnx.load(str(model_path))
    model_fp16 = convert_float_to_float16(model, keep_io_types=True)
    onnx.save(model_fp16, str(onnx_fp16_output / "model.onnx"))

    tokenizer.save_pretrained(str(onnx_fp16_output))
    for f in ["config.json", "special_tokens_map.json", "tokenizer_config.json", "tokenizer.json"]:
        src = model_path.parent / f
        if src.exists():
            shutil.copy(src, onnx_fp16_output / f)

    print(f"FP16 半精度: {onnx_fp16_output}")


def test_model(onnx_dir: Path, mode: str):
    model_file = "model_quantized.onnx" if mode in ("static", "dynamic") else "model.onnx"
    if not (onnx_dir / model_file).exists():
        model_file = "model.onnx"

    try:
        model = ORTModelForTokenClassification.from_pretrained(str(onnx_dir), file_name=model_file)
        tokenizer = AutoTokenizer.from_pretrained(str(onnx_dir))
        ner_pipe = pipeline(
            "token-classification",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="simple",
            device="cpu",
        )
        results = ner_pipe("张三今天在北京开会，见到了李四。")
        for r in results:
            print(f"  {r['word']} -> {r['entity_group']} ({r['score']:.4f})")
    except Exception as e:
        print(f"  测试失败: {e}")


def export_onnx(modes):
    if "all" in modes:
        modes = ["static", "fp16"]

    model_dir = resolve_path(MODEL_DIR)
    onnx_output = resolve_path(ONNX_OUTPUT_DIR)
    onnx_int8_output = resolve_path(ONNX_INT8_OUTPUT_DIR)
    onnx_fp16_output = resolve_path(ONNX_FP16_OUTPUT_DIR)
    vocab_output = resolve_path(VOCAB_OUTPUT)

    if not model_dir.exists():
        raise FileNotFoundError(f"模型目录不存在: {model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    ort_model = ORTModelForTokenClassification.from_pretrained(str(model_dir), export=True)

    tokenizer.save_pretrained(str(onnx_output))
    ort_model.save_pretrained(str(onnx_output))
    export_vocab(tokenizer, str(vocab_output))

    orig_size = (onnx_output / "model.onnx").stat().st_size / 1024 / 1024
    print(f"ONNX 原始模型: {orig_size:.2f} MB -> {onnx_output}")

    quantizer = ORTQuantizer.from_pretrained(ort_model)

    for mode in modes:
        if mode == "static":
            export_static_quantized(quantizer, tokenizer, onnx_int8_output)
            model_file = onnx_int8_output / "model_quantized.onnx"
        elif mode == "fp16":
            export_fp16(ort_model, tokenizer, onnx_fp16_output)
            model_file = onnx_fp16_output / "model.onnx"
        else:
            continue

        if model_file.exists():
            size = model_file.stat().st_size / 1024 / 1024
            print(f"  压缩比: {orig_size / size:.2f}x ({size:.2f} MB)")
            test_model(model_file.parent, mode)


def analyze_ops(model_path: str):
    model = onnx.load(model_path)
    ops = {n.op_type for n in model.graph.node}

    limited = ops & {"MatMulInteger", "DynamicQuantizeLinear", "DequantizeLinear"}
    supported = ops & {"QLinearMatMul", "QLinearAdd", "QLinearConv", "MatMul", "Add", "Mul"}

    print(f"算子类型: {len(ops)}")
    print(f"CUDA 支持: {supported or '标准算子'}")
    if limited:
        print(f"CUDA 受限: {limited}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ONNX 模型导出与量化")
    parser.add_argument(
        "--mode", "-m",
        nargs="+",
        choices=["static", "fp16", "all"],
        default=["static"],
        help="量化模式",
    )
    parser.add_argument("--all", "-a", action="store_true", help="导出全部")
    parser.add_argument("--analyze", action="store_true", help="分析算子")

    args = parser.parse_args()

    if args.analyze:
        model_path = resolve_path(ONNX_INT8_OUTPUT_DIR) / "model_quantized.onnx"
        analyze_ops(str(model_path)) if model_path.exists() else print("模型不存在")
    else:
        export_onnx(["all"] if args.all else args.mode)
