# NER-Gateway

<div align="center">

**Rust 中文姓名识别服务**

![NER-Gateway](https://socialify.git.ci/xihan123/ner-gateway/image?description=1&forks=1&issues=1&language=1&name=1&owner=1&pulls=1&stargazers=1&theme=Auto)

[![CI/CD](https://github.com/xihan123/ner-gateway/actions/workflows/build-release.yml/badge.svg)](https://github.com/xihan123/ner-gateway/actions/workflows/build-release.yml)
[![Release](https://img.shields.io/github/v/release/xihan123/ner-gateway)](https://github.com/xihan123/ner-gateway/releases)
[![License: GPL-3.0](https://img.shields.io/badge/License-GPL--3.0-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

[下载](#下载) • [快速开始](#快速开始) • [编译](#从源码编译) • [API](#api)

</div>

---

## 简介

基于 BERT 的中文姓名识别系统，Rust 后端 + Python 训练管道，ONNX Runtime 推理。

**特点**：

- RTX 4070S 单次推理 32 条文本约 2ms（FP16）
- 自动去重 + 人工审核
- GPU 加速（CUDA 12/13.x）
- 导出 BIO 格式训练数据
- Docker 一键部署

---

## 模型性能

RTX 4070S 12G，真实业务数据测试：

| 模型                  | F1     | Precision | Recall | 时延 (ms) | 吞吐 (条/s) |
|---------------------|--------|-----------|--------|---------|----------|
| onnx_ner_model_fp16 | 0.9998 | 0.9998    | 0.9998 | 0.08    | 13053    |
| onnx_ner_model      | 0.9998 | 0.9998    | 0.9998 | 0.09    | 10793    |
| onnx_ner_model_int8 | 0.9998 | 0.9999    | 0.9998 | 0.19    | 5331     |

**量化对比**：

| 格式   | 大小     | GPU | 精度 | 速度     |
|------|--------|-----|----|--------|
| FP32 | ~140MB | ✓   | 最高 | 基准     |
| FP16 | ~70MB  | ✓   | 高  | 1.5-2x |
| INT8 | ~35MB  | ✓   | 中  | 2-3x   |

---

## 下载

从 [GitHub Releases](https://github.com/xihan123/ner-gateway/releases/latest) 下载。

| 平台          | 文件                            |
|-------------|-------------------------------|
| Windows x64 | `ner-gateway-windows-x64.exe` |
| macOS ARM   | `ner-gateway-macos-arm64`     |
| Linux x64   | `ner-gateway-linux-x64`       |

需要模型文件和词表（`models/` 目录）。

---

## 快速开始

### 本地运行

```bash
set NER_MODEL_PATH=./models/onnx_ner_model_fp16/model.onnx
set NER_VOCAB_PATH=./models/onnx_ner_model_fp16/vocab.txt
ner-gateway-windows-x64.exe
```

### Docker

```bash
docker-compose up -d --build
```

访问 <http://localhost:8080> 打开审核界面。

### 测试

```powershell
curl -X POST http://localhost:8080/api/extract `
  -H "Content-Type: application/json" `
  -d '{\"text\": \"张三和李四在北京参加会议\"}'
```

```json
{
  "names": ["张三", "李四"],
  "confidence": 0.95,
  "review_id": 1,
  "is_duplicate": false
}
```

---

## 从源码编译

### 前置要求

- Rust 1.75+
- Python 3.11.x（训练用）

### 编译

```bash
git clone https://github.com/xihan123/ner-gateway.git
cd ner-gateway
cargo build --release
```

### 跨平台

```bash
# Windows x64
cargo build --release --target x86_64-pc-windows-msvc

# macOS ARM
cargo build --release --target aarch64-apple-darwin

# Linux x64
cargo build --release --target x86_64-unknown-linux-gnu
```

---

## API

| 端点                    | 方法   | 说明                          |
|-----------------------|------|-----------------------------|
| `/api/extract`        | POST | 提取姓名                        |
| `/api/extract/batch`  | POST | 批量提取（上限限 100 条）推荐8、16、32、64 |
| `/api/reviews`        | GET  | 待审核列表                       |
| `/api/reviews/filter` | GET  | 筛选审核数据                      |
| `/api/reviews/:id`    | POST | 更新审核状态                      |
| `/api/stats`          | GET  | 统计信息                        |
| `/api/export`         | GET  | 导出训练数据                      |
| `/api/gpu`            | GET  | GPU 状态                      |
| `/health`             | GET  | 健康检查                        |

### 提取姓名

```bash
curl -X POST http://localhost:8080/api/extract \
  -H "Content-Type: application/json" \
  -d '{"text": "张三在北京工作"}'
```

### 批量提取

```bash
curl -X POST http://localhost:8080/api/extract/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["张三在北京", "李四来自上海"]}'
```

### 筛选审核数据

```bash
curl "http://localhost:8080/api/reviews/filter?status=pending&confidence_min=0.8&has_names=true&limit=100"
```

### 更新审核

```bash
# 批准
curl -X POST http://localhost:8080/api/reviews/1 \
  -H "Content-Type: application/json" \
  -d '{"action": "approve"}'

# 修正
curl -X POST http://localhost:8080/api/reviews/1 \
  -H "Content-Type: application/json" \
  -d '{"action": "correct", "names": ["李四"]}'

# 拒绝
curl -X POST http://localhost:8080/api/reviews/1 \
  -H "Content-Type: application/json" \
  -d '{"action": "reject"}'
```

### 导出数据

```bash
curl "http://localhost:8080/api/export?format=jsonl"
```

---

## 配置

| 变量                 | 默认值                                       | 说明               |
|--------------------|-------------------------------------------|------------------|
| `NER_MODEL_PATH`   | `./models/onnx_ner_model_fp16/model.onnx` | 模型路径             |
| `NER_VOCAB_PATH`   | `./models/onnx_ner_model_fp16/vocab.txt`  | 词表路径             |
| `NER_DB_PATH`      | `./ner_reviews.db`                        | 数据库路径            |
| `NER_PORT`         | `8080`                                    | 服务端口             |
| `NER_DISABLE_CUDA` | -                                         | 设为 `true` 禁用 GPU |
| `OPENAI_API_KEY`   | -                                         | 大模型 API 密钥       |
| `OPENAI_BASE_URL`  | -                                         | API 地址           |
| `OPENAI_MODEL`     | `deepseek-chat`                           | 模型名称             |

---

## GPU 加速

### 要求

- NVIDIA GPU 4G (实测训练只占用3G左右，训练十分钟)
- CUDA 12.x / 13.x
- cuDNN 9.x

### Windows 配置

自动检测以下路径：

```
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64
C:\Program Files\NVIDIA\CUDNN\v9.19\bin\13.1\x64
```

### 注意

- 推荐 FP16 模型获得最佳性能
- 加载失败自动回退 CPU

---

## 训练

### 完整流程

```bash
# 1. 安装依赖
uv sync

# 2. 生成数据（可选）
set OPENAI_API_KEY=sk-xxx
set OPENAI_BASE_URL=https://api.deepseek.com
uv run python training/generate_data.py -o raw_data/training_data.jsonl -c 1000

# 3. 生成负样本（可选）
uv run python training/generate_negative_samples.py

# 4. 数据清洗与划分
uv run python training/split_dataset.py

# 5. 训练
uv run python training/train.py

# 6. 导出 ONNX
uv run python training/export_onnx.py --mode all
```

### 数据增强

```bash
# API 增强
uv run python training/augment_data.py -i raw_data/training_data.jsonl -o raw_data/augmented.jsonl -c 500

# 简单增强（不调 API）
uv run python training/augment_data.py -i raw_data/training_data.jsonl -o raw_data/augmented.jsonl --simple-only
```

### 基准测试

```bash
uv run python training/benchmark.py -m models/onnx_ner_model_fp16 --onnx
uv run python training/benchmark.py -m models/best_ner_model -m models/onnx_ner_model_int8 --onnx
```

---

## 项目结构

```
backend/src/
├── main.rs      # 入口与路由
├── engine.rs    # ONNX 推理引擎
├── tokenizer.rs # BERT 分词器
├── db.rs        # 数据库操作
├── handlers.rs  # API 处理器
└── config.rs    # 配置

training/
├── train.py                   # 训练
├── export_onnx.py             # ONNX 导出
├── benchmark.py               # 基准测试
├── generate_data.py           # 数据生成
├── generate_negative_samples.py  # 负样本
├── augment_data.py            # 数据增强
├── split_dataset.py           # 数据划分
└── test_onnx.py               # 模型测试

models/
├── onnx_ner_model/            # FP32
├── onnx_ner_model_fp16/       # FP16（推荐）
├── onnx_ner_model_int8/       # INT8
└── best_ner_model/            # 训练输出
```

---

## 数据飞轮

```
推理 → 存储 → 审核 → 导出 → 重训练 → 部署
```

1. `/api/extract` 识别姓名
2. 结果存入 SQLite（SHA256 去重）
3. 前端人工审核
4. 导出训练数据
5. 重训练并部署

---

## 致谢

- [axum](https://github.com/tokio-rs/axum) - Web 框架
- [ort](https://github.com/pykeio/ort) - ONNX Runtime 绑定
- [rusqlite](https://github.com/rusqlite/rusqlite) - SQLite 绑定
- [transformers](https://github.com/huggingface/transformers) - 模型训练
- [optimum](https://github.com/huggingface/optimum) - ONNX 导出

---

<div align="center">

**有帮助？给个 Star**

[⬆ 返回顶部](#ner-gateway)

</div>
