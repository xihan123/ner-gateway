import json


def generate_bio_tags(text, entities):
    """
    将文本和实体列表转换为 BIO 标注序列
    B-PER: 人名首字
    I-PER: 人名非首字
    O: 非人名
    """
    # 初始化全为 'O'
    labels = ["O"] * len(text)

    # 按照实体长度降序排序，防止子串匹配错误（如先匹配"张三"，再匹配"张三丰"会有问题）
    entities = sorted(entities, key=len, reverse=True)

    # 标记已经被占用的位置，防止重叠匹配
    occupied = [False] * len(text)

    for entity in entities:
        if not entity:
            continue
        entity_len = len(entity)

        # 在文本中查找所有实体的出现位置
        start = 0
        while True:
            idx = text.find(entity, start)
            if idx == -1:
                break

            # 检查这个位置是否已经被其他实体占用
            is_overlap = any(occupied[idx : idx + entity_len])

            if not is_overlap:
                # 标记为 B-PER 和 I-PER
                labels[idx] = "B-PER"
                for i in range(1, entity_len):
                    labels[idx + i] = "I-PER"
                # 更新占用标记
                for i in range(entity_len):
                    occupied[idx + i] = True

            start = idx + 1

    return labels


def process_ai_generated_data(input_file, output_file):
    """处理大模型生成的 jsonl 文件"""
    processed_count = 0
    with open(input_file, "r", encoding="utf-8") as fin, open(
        output_file, "w", encoding="utf-8"
    ) as fout:

        for line in fin:
            if not line.strip():
                continue
            try:
                data = json.loads(line.strip())
                text = data["text"]
                entities = data.get("entities", [])

                # 生成 BIO 标签
                labels = generate_bio_tags(text, entities)

                # 验证长度一致性
                if len(text) == len(labels):
                    # 组装为训练模型需要的格式
                    output_data = {
                        "tokens": list(text),  # 拆分为单字列表，符合 BERT 习惯
                        "labels": labels,
                    }
                    fout.write(json.dumps(output_data, ensure_ascii=False) + "\n")
                    processed_count += 1
                else:
                    print(f"长度不匹配，跳过: {text}")

            except Exception as e:
                print(f"解析错误: {line.strip()} - {e}")

    print(
        f"✅ 处理完成！成功生成 {processed_count} 条标准 BIO 训练数据，保存在 {output_file}"
    )


# ================= 运行示例 =================
if __name__ == "__main__":
    # 假设你把大模型生成的文本保存成了 raw_ai_data.jsonl
    # process_ai_generated_data('raw_ai_data.jsonl', 'train_dataset.jsonl')

    # 演示单个句子的转换结果
    sample_text = "你让张三李四马上到我办公室，顺便叫上欧阳修和那个叫约翰的客户。"
    sample_entities = ["张三", "李四", "欧阳修", "约翰"]

    tags = generate_bio_tags(sample_text, sample_entities)

    print("文本:", list(sample_text))
    print("标签:", tags)
