# Changelog

## [1.4.1](https://github.com/xihan123/ner-gateway/compare/v1.4.0...v1.4.1) (2026-03-02)


### Bug Fixes

* **config:** 更新模型路径配置 ([64d6431](https://github.com/xihan123/ner-gateway/commit/64d64313e83316895d84b9c12e34869a4975eaa7))

## [1.4.0](https://github.com/xihan123/ner-gateway/compare/v1.3.0...v1.4.0) (2026-03-02)


### Features

* **ai_review:** 添加AI审核脚本用于批量审核姓名识别结果 ([425c695](https://github.com/xihan123/ner-gateway/commit/425c695c5c4bf947e97529f244514cccdda3ea6f))
* **api:** 添加批量提取功能并优化API处理器 ([ee3606a](https://github.com/xihan123/ner-gateway/commit/ee3606ada42a093cb9ae0c2c958c47ddffd9d291))
* **dataset:** 添加数据清洗与划分脚本 ([5b6f21f](https://github.com/xihan123/ner-gateway/commit/5b6f21f9c2c1c6fca34750f6aabb0be013aeb36b))
* **data:** 添加数据清洗功能并优化 BIO 标签转换器 ([ee55bbe](https://github.com/xihan123/ner-gateway/commit/ee55bbec11b54c8ba91b24df8563ef8e2f40ccb1))
* **dependencies:** 添加ONNX相关依赖包 ([19007f4](https://github.com/xihan123/ner-gateway/commit/19007f43dc4e51ec3fa5b495532a276026871970))
* **engine:** 添加GPU支持和性能优化 ([8d720e6](https://github.com/xihan123/ner-gateway/commit/8d720e6e3fc10cfea956541d66257b549c58d9bf))
* **model:** 添加 CUDA 支持以加速模型推理 ([4845b9c](https://github.com/xihan123/ner-gateway/commit/4845b9c582cc92d397f8cbac29bc1665cdabc8db))
* **nlp:** 添加 NER 模型统一基准测试脚本 ([ed73b94](https://github.com/xihan123/ner-gateway/commit/ed73b943a10bbe5cf301f83fcef735b06c7876d6))
* **training:** 添加 JSONL 文件合并工具 ([58c6a58](https://github.com/xihan123/ner-gateway/commit/58c6a58c9b39a530ee0123e1557fb782073f8e43))
* **training:** 添加数据增强脚本用于中文姓名识别训练 ([f75b0e7](https://github.com/xihan123/ner-gateway/commit/f75b0e7f5a38fdd4cbd6ee17716fefb991fb3b1c))


### Performance Improvements

* **db:** 优化数据库性能并添加哈希缓存 ([96b58ee](https://github.com/xihan123/ner-gateway/commit/96b58eec36fd378ef0ca277b798d0f5edc4dab4a))

## [1.3.0](https://github.com/xihan123/ner-gateway/compare/v1.2.0...v1.3.0) (2026-02-27)


### Features

* **engine:** 添加姓名验证和清理功能以提高实体提取准确性 ([289343d](https://github.com/xihan123/ner-gateway/commit/289343d29d9492a40fd44eb64e4f8df33d5e9110))

## [1.2.0](https://github.com/xihan123/ner-gateway/compare/v1.1.0...v1.2.0) (2026-02-26)


### Features

* **db:** 添加高级筛选、分页功能、统计查询支持 ([ffc608a](https://github.com/xihan123/ner-gateway/commit/ffc608a15efccc05f94f1e7487f144429b74a997))

## [1.1.0](https://github.com/xihan123/ner-gateway/compare/v1.0.0...v1.1.0) (2026-02-25)


### Features

* **core:** 初始化 NER 网关核心功能 ([5265a4b](https://github.com/xihan123/ner-gateway/commit/5265a4b5928bf4e1cfcce634a3cfeadec986c739))


### Bug Fixes

* **ci:** 修复Docker镜像构建工作流权限配置 ([23532de](https://github.com/xihan123/ner-gateway/commit/23532dedbffb5184e68157b9ca3749a3186161fa))

## 1.0.0 (2026-02-25)


### Features

* **core:** 初始化 NER 网关核心功能 ([5265a4b](https://github.com/xihan123/ner-gateway/commit/5265a4b5928bf4e1cfcce634a3cfeadec986c739))
