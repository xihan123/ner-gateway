#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
大模型数据生成器 (generate_data.py)

功能：
    调用 OpenAI 兼容 API（支持 DeepSeek、Qwen 等），生成客服场景的仿真语料。
    生成的数据涵盖多种姓名提取场景，包括：
    - 常规中文姓名
    - 尊称夹杂（如"张总"、"Tony老师"）
    - 纯干扰项（如"张三丰满的衣服"、"平安客服"）
    - 生僻字姓名
    - 复姓（如"欧阳"、"司马"）
    - 英文名/中英混合

输出格式：
    JSONL 文件，每行格式：{"text": "原始文本", "entities": ["名字1", "名字2", ...]}

使用方法：
    python generate_data.py --output raw_ai_data.jsonl --count 50

环境变量：
    OPENAI_API_KEY: API 密钥（必需）
    OPENAI_BASE_URL: API 基础 URL（可选，用于兼容 DeepSeek 等服务）
    OPENAI_MODEL: 模型名称（可选，默认 deepseek-chat）
"""

import json
import os
import argparse
import time
from typing import List, Dict, Any
from openai import OpenAI


# ============================================================
# Prompt 模板：指导大模型生成客服场景语料
# ============================================================

SYSTEM_PROMPT = """你是一个客服对话数据生成专家。你需要生成逼真的客服聊天场景文本，并标注其中的人名实体。

【标注规则】
1. 只标注真实的"人名"，包括：中文姓名、英文名、中英混合名
2. 尊称/职称/关系词需要剥离：
   - "张总" → 只提取"张"（"总"是职务称呼）
   - "Tony老师" → 只提取"Tony"（"老师"是职业称呼）
   - "李哥" → 只提取"李"（"哥"是关系称呼）
   - "王女士" → 只提取"王"（"女士"是性别称呼）
   - "刘经理" → 只提取"刘"（"经理"是职务）
   - "陈医生" → 只提取"陈"（"医生"是职业）
3. 干扰项不提取：
   - "张三丰满的衣服" → "张三丰"是完整人名，但"张三丰满"中的"满"属于句子的一部分
   - "平安客服" → 无姓名，不提取
   - "您的问题已解决" → 无姓名，不提取
4. 复姓处理：
   - 欧阳、司马、诸葛、上官、皇甫等复姓，完整标注：如"欧阳锋"标注为完整实体
5. 生僻字姓名正常标注

【场景类型】
生成以下类型的客服对话文本（每条独立，非连续对话）：
- 咨询类：用户咨询问题
- 投诉类：用户投诉反馈
- 售后类：退换货、维修
- 预约类：预约服务
- 投诉处理：客服回复用户
- 内部沟通：客服人员之间交流

【输出格式】
严格按以下 JSON 格式输出，不要添加任何其他内容：
{{"text": "对话文本内容", "entities": ["人名1", "人名2"]}}"""

USER_PROMPT_TEMPLATE = """请生成 {batch_size} 条客服场景文本，要求：
1. 每条文本长度 15-80 字
2. 约 30% 的文本包含 1 个姓名
3. 约 20% 的文本包含 2-3 个姓名
4. 约 20% 的文本包含尊称/职称（需要剥离的情况）
5. 约 15% 的文本包含干扰项或无姓名
6. 约 15% 的文本包含生僻字/复姓/英文名

请直接输出 JSON 数组格式：
[
  {{"text": "...", "entities": [...]}},
  ...
]"""


def create_client() -> OpenAI:
    """
    创建 OpenAI 客户端，支持兼容 API 服务。
    
    Returns:
        OpenAI 客户端实例
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("请设置环境变量 OPENAI_API_KEY")
    
    # 支持自定义 base_url（DeepSeek、Qwen 等）
    base_url = os.environ.get("OPENAI_BASE_URL")
    
    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    
    return OpenAI(**client_kwargs)


def generate_batch(
    client: OpenAI,
    model: str,
    batch_size: int = 10,
    max_retries: int = 3
) -> List[Dict[str, Any]]:
    """
    调用 API 生成一批数据。
    
    Args:
        client: OpenAI 客户端
        model: 模型名称
        batch_size: 每批生成的数量
        max_retries: 最大重试次数
    
    Returns:
        生成的数据列表
    """
    user_prompt = USER_PROMPT_TEMPLATE.format(batch_size=batch_size)
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=1.5,  # 增加多样性
                max_tokens=4000,
            )
            
            content = response.choices[0].message.content.strip()
            
            # 尝试解析 JSON 响应
            # 处理可能的 markdown 代码块包裹
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
            
            data = json.loads(content)
            
            # 验证数据格式
            if isinstance(data, list):
                validated = []
                for item in data:
                    if isinstance(item, dict) and "text" in item and "entities" in item:
                        # 确保 entities 是列表
                        if isinstance(item["entities"], str):
                            item["entities"] = [item["entities"]]
                        validated.append(item)
                return validated
            
        except json.JSONDecodeError as e:
            print(f"[警告] JSON 解析失败 (尝试 {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
        except Exception as e:
            print(f"[错误] API 调用失败 (尝试 {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
    
    return []


def generate_dataset(
    total_count: int = 50,
    output_path: str = "raw_ai_data.jsonl",
    batch_size: int = 10,
    model: str = None,
    append: bool = False
) -> int:
    """
    生成完整数据集并保存到文件。
    
    Args:
        total_count: 总数据条数
        output_path: 输出文件路径
        batch_size: 每批生成的数量
        model: 使用的模型名称
        append: 是否追加到现有文件
    
    Returns:
        实际生成的数据条数
    """
    # 创建客户端
    client = create_client()
    
    # 获取模型名称
    if model is None:
        model = os.environ.get("OPENAI_MODEL", "deepseek-chat")
    
    # 统计现有数据条数
    existing_count = 0
    if append and os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            existing_count = sum(1 for line in f if line.strip())
        print(f"[信息] 追加模式：现有 {existing_count} 条数据")
    
    print(f"[信息] 开始生成数据，目标数量: {total_count}")
    print(f"[信息] 使用模型: {model}")
    print(f"[信息] 输出文件: {output_path}")
    print("-" * 50)
    
    total_generated = 0
    batches_needed = (total_count + batch_size - 1) // batch_size
    
    for batch_idx in range(batches_needed):
        remaining = total_count - total_generated
        current_batch_size = min(batch_size, remaining)
        
        print(f"[进度] 正在生成第 {batch_idx + 1}/{batches_needed} 批...")
        
        batch_data = generate_batch(client, model, current_batch_size)
        
        if batch_data:
            # 截断到剩余需要的数量
            batch_data = batch_data[:remaining]
            
            # 立即写入文件（第一批根据 append 参数决定模式，后续批次追加）
            if batch_idx == 0 and not append:
                write_mode = "w"
            else:
                write_mode = "a"
            
            with open(output_path, write_mode, encoding="utf-8") as f:
                for item in batch_data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
            total_generated += len(batch_data)
            print(f"[成功] 本批生成并保存 {len(batch_data)} 条，累计 {total_generated} 条")
        else:
            print(f"[失败] 本批生成失败，跳过")
        
        # 达到目标数量后停止
        if total_generated >= total_count:
            break
        
        # 避免请求过快
        if batch_idx < batches_needed - 1:
            time.sleep(1)
    
    print("-" * 50)
    
    # 统计最终文件条数
    with open(output_path, "r", encoding="utf-8") as f:
        final_count = sum(1 for line in f if line.strip())
    
    print(f"[完成] 本次生成 {total_generated} 条数据，已保存至 {output_path}")
    print(f"[统计] 文件现有共 {final_count} 条数据")
    
    return total_generated


def main():
    """主函数入口。"""
    parser = argparse.ArgumentParser(
        description="客服场景姓名实体数据生成器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python generate_data.py --output raw_ai_data.jsonl --count 50
    python generate_data.py -o raw_ai_data.jsonl -c 100 --append  # 追加模式
    
环境变量:
    OPENAI_API_KEY   API 密钥（必需）
    OPENAI_BASE_URL  API 基础 URL（可选，用于 DeepSeek 等服务）
    OPENAI_MODEL     模型名称（可选，默认 deepseek-chat）

DeepSeek 示例:
    set OPENAI_API_KEY=sk-xxx
    set OPENAI_BASE_URL=https://api.deepseek.com
    set OPENAI_MODEL=deepseek-chat
    python generate_data.py
        """
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="raw_ai_data.jsonl",
        help="输出 JSONL 文件路径 (默认: raw_ai_data.jsonl)"
    )
    
    parser.add_argument(
        "--count", "-c",
        type=int,
        default=1000,
        help="生成数据条数 (默认: 50)"
    )
    
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=100,
        help="每批生成数量 (默认: 10)"
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="模型名称 (默认: 从环境变量 OPENAI_MODEL 读取，或 deepseek-chat)"
    )
    
    parser.add_argument(
        "--append", "-a",
        action="store_true",
        help="追加到现有文件而非覆盖"
    )
    
    args = parser.parse_args()
    
    # 执行生成
    generate_dataset(
        total_count=args.count,
        output_path=args.output,
        batch_size=args.batch_size,
        model=args.model,
        append=args.append
    )


if __name__ == "__main__":
    main()
