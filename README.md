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

基于 BERT 的中文姓名实体识别系统，使用 ONNX Runtime 进行高效推理。Rust 后端 + Python 训练管道，支持数据飞轮持续学习。

核心特性：
- 单次推理延迟 < 10ms（INT8 量化模型）
- 自动去重 + 人工审核流程
- 导出 BIO 格式训练数据
- Docker 一键部署

---

## 下载

从 [GitHub Releases](https://github.com/xihan123/ner-gateway/releases/latest) 下载。

| 平台 | 文件名 |
|------|--------|
| Windows x64 | `ner-gateway-windows-x64.exe` |
| macOS Intel | `ner-gateway-macos-x64` |
| macOS Apple Silicon | `ner-gateway-macos-arm64` |
| Linux x64 | `ner-gateway-linux-x64` |

需要模型文件 `ner_model_int8.onnx` 和 `vocab.txt`。

---

## 快速开始

### 本地运行

```bash
# 设置模型路径
set NER_MODEL_PATH=./models/ner_model_int8.onnx
set NER_VOCAB_PATH=./models/vocab.txt

# 运行服务
ner-gateway-windows-x64.exe
```

### Docker

```bash
docker-compose up -d --build
```

服务启动后访问 http://localhost:8080 打开审核界面。

### 测试接口

```powershell
curl -X POST http://localhost:8080/api/extract `
  -H "Content-Type: application/json" `
  -d '{\"text\": \"张三和李四在北京参加会议\"}'
```

响应：

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
- Python 3.11+（训练用）
- Git

### 编译后端

```bash
git clone https://github.com/xihan123/ner-gateway.git
cd ner-gateway
cargo build --release
```

### 跨平台编译

```bash
# Windows x64
cargo build --release --target x86_64-pc-windows-msvc

# macOS Intel
cargo build --release --target x86_64-apple-darwin

# macOS Apple Silicon
cargo build --release --target aarch64-apple-darwin

# Linux x64
cargo build --release --target x86_64-unknown-linux-gnu
```

---

## API

| 端点 | 方法 | 描述 |
|------|------|------|
| `/api/extract` | POST | 提取姓名 |
| `/api/reviews` | GET | 获取待审核列表 |
| `/api/reviews/all` | GET | 获取所有记录 |
| `/api/reviews/:id` | POST | 更新审核状态 |
| `/api/stats` | GET | 统计信息 |
| `/api/export` | GET | 导出训练数据 |
| `/health` | GET | 健康检查 |

### 提取姓名

```bash
curl -X POST http://localhost:8080/api/extract \
  -H "Content-Type: application/json" \
  -d '{"text": "张三在北京工作"}'
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
# JSON 格式
curl http://localhost:8080/api/export

# JSONL 格式（BIO 标注）
curl "http://localhost:8080/api/export?format=jsonl"
```

---

## 配置

环境变量：

| 变量 | 默认值 | 描述 |
|------|--------|------|
| `NER_MODEL_PATH` | `./models/ner_model_int8.onnx` | ONNX 模型路径 |
| `NER_VOCAB_PATH` | `./models/vocab.txt` | 词表路径 |
| `NER_DB_PATH` | `./ner_reviews.db` | 数据库路径 |
| `NER_PORT` | `8080` | 服务端口 |

---

## 训练

```bash
# 安装依赖
uv sync

# 训练模型
uv run python training/train.py

# 导出 ONNX
uv run python training/export_onnx.py

# 生成训练数据
set OPENAI_API_KEY=sk-xxx
uv run python training/generate_data.py -o data/raw_ai_data.jsonl -c 1000
```

---

## 开发

```bash
# 运行测试
cargo test

# 运行后端
cd backend
cargo run --release

# Python 测试
uv run python training/test_onnx.py
```

项目结构：

```
backend/
├── src/
│   ├── main.rs      # 入口
│   ├── engine.rs    # ONNX 推理引擎
│   ├── tokenizer.rs # BERT 分词器
│   ├── db.rs        # 数据库操作
│   ├── handlers.rs  # API 处理器
│   └── config.rs    # 配置

training/
├── train.py         # 模型训练
├── export_onnx.py   # ONNX 导出
├── generate_data.py # 数据生成
└── test_onnx.py     # 模型测试
```

---

## 数据飞轮

```
推理 → 存储 → 审核 → 导出 → 重训练 → 部署
```

1. 调用 `/api/extract` 识别姓名
2. 结果自动存入 SQLite（SHA256 去重）
3. 前端界面人工审核（确认/修正/拒绝）
4. 导出审核数据作为训练集
5. 重新训练模型并部署

---

## 致谢

主要依赖：

- [axum](https://github.com/tokio-rs/axum) - Web 框架
- [ort](https://github.com/pykeio/ort) - ONNX Runtime 绑定
- [rusqlite](https://github.com/rusqlite/rusqlite) - SQLite 绑定
- [transformers](https://github.com/huggingface/transformers) - 模型训练
- [Vue 3](https://github.com/vuejs/vue) - 前端界面

---

<div align="center">

**如果这个项目对您有帮助，请给一个 Star**

[⬆ 返回顶部](#ner-gateway)

</div>
