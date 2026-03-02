#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any


def load_jsonl(file_path: Path) -> list[dict[str, Any]]:
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"警告: {file_path} 第 {line_no} 行 JSON 解析失败: {e}")
    return data


def save_jsonl(data: list[dict[str, Any]], file_path: Path) -> None:
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def tokens_to_key(tokens: list[str]) -> str:
    return "\x00".join(tokens)


def merge_jsonl(
    base_path: Path,
    update_path: Path,
    output_path: Path,
    verbose: bool = False,
) -> dict[str, int]:
    base_data = load_jsonl(base_path)
    update_data = load_jsonl(update_path)

    base_order: list[str] = []
    base_map: dict[str, dict[str, Any]] = {}

    for item in base_data:
        if "tokens" not in item:
            continue
        key = tokens_to_key(item["tokens"])
        if key not in base_map:
            base_order.append(key)
        base_map[key] = item

    update_map: dict[str, dict[str, Any]] = {}
    for item in update_data:
        if "tokens" not in item:
            continue
        key = tokens_to_key(item["tokens"])
        update_map[key] = item

    replaced = 0
    added = 0

    result: list[dict[str, Any]] = []
    for key in base_order:
        if key in update_map:
            result.append(update_map[key])
            replaced += 1
        else:
            result.append(base_map[key])

    for key, item in update_map.items():
        if key not in base_map:
            result.append(item)
            added += 1

    save_jsonl(result, output_path)

    return {
        "base_count": len(base_data),
        "update_count": len(update_data),
        "replaced": replaced,
        "added": added,
        "output_count": len(result),
    }


def main():
    parser = argparse.ArgumentParser(description="合并两个 JSONL 文件，以 tokens 为 key 进行匹配替换")
    parser.add_argument("-b", "--base", required=True, help="基础 JSONL 文件路径")
    parser.add_argument("-u", "--update", required=True, help="更新 JSONL 文件路径")
    parser.add_argument("-o", "--output", help="输出 JSONL 文件路径（默认覆盖 base 文件）")
    parser.add_argument("-v", "--verbose", action="store_true", help="显示详细统计信息")

    args = parser.parse_args()

    base_path = Path(args.base)
    update_path = Path(args.update)
    output_path = Path(args.output) if args.output else base_path

    if not base_path.exists():
        print(f"错误: 基础文件不存在: {base_path}")
        return 1
    if not update_path.exists():
        print(f"错误: 更新文件不存在: {update_path}")
        return 1

    stats = merge_jsonl(base_path, update_path, output_path, args.verbose)

    if args.verbose:
        print(f"基础文件条目数: {stats['base_count']}")
        print(f"更新文件条目数: {stats['update_count']}")
        print(f"替换条目数: {stats['replaced']}")
        print(f"新增条目数: {stats['added']}")
        print(f"输出文件条目数: {stats['output_count']}")
        print(f"输出文件: {output_path}")
    else:
        print(f"合并完成: 替换 {stats['replaced']} 条, 新增 {stats['added']} 条, 共 {stats['output_count']} 条")

    return 0


if __name__ == "__main__":
    exit(main())