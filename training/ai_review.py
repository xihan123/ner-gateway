#!/usr/bin/env python
"""AI 辅助审核姓名识别结果。"""

import argparse
import json
import os
import sqlite3
import time
import urllib.error
import urllib.request
from datetime import datetime
from typing import Any, Dict, List

from openai import OpenAI

REVIEW_SYSTEM_PROMPT = """你是一个专业的中文姓名识别审核专家。你的任务是批量审核机器学习模型识别出的姓名是否正确。

【审核规则】
1. 只保留真实的"人名"，包括：中文姓名、英文名、中英混合名
2. 尊称/职称/关系词需要剥离：
   - "张总" → 只保留"张"（"总"是职务称呼）
   - "Tony老师" → 只保留"Tony"（"老师"是职业称呼）
   - "李哥" → 只保留"李"（"哥"是关系称呼）
   - "王女士" → 只保留"王"（"女士"是性别称呼）
3. 客服相关名称不提取：
   - "客服小美"、"小美客服" → 不是用户姓名，是客服昵称
   - "客服001"、"工号123" → 不是人名
   - "小冰"、"小微"等 AI 客服名称 → 不是人名
   - "平安客服"、"招商客服" → 不是人名
   - 判断标准：客服是服务提供方，不是被服务的用户
4. 干扰项不提取：
   - "张三丰满的衣服" → "张三丰"是人名，但要确认上下文
   - 地名、机构名、产品名都不是人名
5. 复姓处理：
   - 欧阳、司马、诸葛、上官、皇甫等复姓，完整保留
6. 生僻字姓名正常识别

【输出格式】
严格按以下 JSON 数组格式输出，每条记录包含：
{
  "id": 记录ID,
  "verified_names": ["审核后的人名列表"],
  "action": "approve/correct",
  "reason": "简要说明"
}

action 说明：
- approve: 模型识别结果完全正确
- correct: 模型识别结果需要修正
  - 提取错误（非人名、客服昵称）→ verified_names 设为空数组 []
  - 需要修正姓名 → 提供修正后的名字列表
  - 需要剥离尊称 → 提供剥离后的名字
"""

BATCH_USER_PROMPT = """请批量审核以下 {count} 条姓名识别结果，按顺序返回审核结果：

{items}

输出格式：
[
  {{"id": 1, "verified_names": [...], "action": "approve/correct", "reason": "..."}},
  ...
]
"""


def create_client() -> OpenAI:
	api_key = os.environ.get("OPENAI_API_KEY")
	if not api_key:
		raise ValueError("请设置环境变量 OPENAI_API_KEY")

	base_url = os.environ.get("OPENAI_BASE_URL")
	client_kwargs = {"api_key": api_key}
	if base_url:
		client_kwargs["base_url"] = base_url

	return OpenAI(**client_kwargs)


def get_db_path() -> str:
	return "../ner_reviews.db"


def get_api_url() -> str:
	return os.environ.get("NER_API_URL", "http://localhost:8080")


def get_pending_reviews(db_path: str, limit: int) -> List[Dict[str, Any]]:
	"""
	从数据库获取待审核数据。

	Args:
		db_path: 数据库路径
		limit: 获取数量

	Returns:
		待审核数据列表
	"""
	conn = sqlite3.connect(db_path)
	conn.row_factory = sqlite3.Row
	cursor = conn.cursor()

	cursor.execute("""
                   SELECT id, original_text, predicted_names, confidence, created_at
                   FROM review_data
                   WHERE status = 'pending'
                     AND predicted_names IS NOT NULL
                     AND predicted_names != '[]'
                   ORDER BY created_at DESC
                   LIMIT ?
	               """, (limit,))

	rows = cursor.fetchall()
	conn.close()

	reviews = []
	for row in rows:
		predicted_names = json.loads(row["predicted_names"]) if row["predicted_names"] else []
		reviews.append({
			"id": row["id"],
			"original_text": row["original_text"],
			"predicted_names": predicted_names,
			"confidence": row["confidence"],
			"created_at": row["created_at"]
		})

	return reviews


def review_batch(
		client: OpenAI,
		model: str,
		review_items: List[Dict[str, Any]],
		max_retries: int = 3
) -> List[Dict[str, Any]]:
	"""
	批量调用 API 审核数据。

	Args:
		client: OpenAI 客户端
		model: 模型名称
		review_items: 待审核数据列表
		max_retries: 最大重试次数

	Returns:
		审核结果列表（与输入顺序对应）
	"""
	if not review_items:
		return []

	items_text = []
	for idx, item in enumerate(review_items):
		items_text.append(f"""[{idx + 1}] ID={item['id']}
原文: {item['original_text']}
预测: {json.dumps(item['predicted_names'], ensure_ascii=False)}
置信度: {item['confidence']}""")

	user_prompt = BATCH_USER_PROMPT.format(
		count=len(review_items),
		items="\n\n".join(items_text)
	)

	for attempt in range(max_retries):
		try:
			response = client.chat.completions.create(
				model=model,
				messages=[
					{"role": "system", "content": REVIEW_SYSTEM_PROMPT},
					{"role": "user", "content": user_prompt}
				],
				temperature=0.1,
				max_tokens=8000,  # 批量输出需要更多 token
			)

			content = response.choices[0].message.content.strip()

			# 处理 markdown 代码块
			if content.startswith("```"):
				lines = content.split("\n")
				content = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])

			results = json.loads(content)

			if isinstance(results, list):
				result_map = {r.get("id"): r for r in results if isinstance(r, dict) and "id" in r}

				ordered_results = []
				for item in review_items:
					result = result_map.get(item["id"])
					if result and "verified_names" in result and "action" in result:
						ordered_results.append(result)
					else:
						ordered_results.append(None)
				return ordered_results
			else:
				print(f"返回格式错误: {str(content)[:100]}")

		except json.JSONDecodeError as e:
			print(f"JSON 解析失败 ({attempt + 1}/{max_retries}): {e}")
			if attempt < max_retries - 1:
				time.sleep(2)
		except Exception as e:
			print(f"API 调用失败 ({attempt + 1}/{max_retries}): {e}")
			if attempt < max_retries - 1:
				time.sleep(3)

	# 失败时返回全 None
	return [None] * len(review_items)


def do_review(count: int, output_path: str, model: str = None, batch_size: int = 20) -> int:
	"""
	执行 AI 审核流程（批量处理）。

	Args:
		count: 审核数量
		output_path: 输出 JSON 文件路径
		model: 使用的模型名称
		batch_size: 每批处理数量（默认 20）

	Returns:
		成功审核的数量
	"""
	# 创建客户端
	client = create_client()

	if model is None:
		model = os.environ.get("OPENAI_MODEL", "deepseek-chat")

	db_path = get_db_path()

	if not os.path.exists(db_path):
		print(f"数据库不存在: {db_path}")
		return 0

	print(f"数据库: {db_path}")
	reviews = get_pending_reviews(db_path, count)

	if not reviews:
		print("没有待审核数据")
		return 0

	print(f"待审核: {len(reviews)} 条 | 模型: {model} | 批量: {batch_size}")
	print(f"输出: {output_path}")

	results = []
	success_count = 0

	total_batches = (len(reviews) + batch_size - 1) // batch_size

	for batch_idx in range(total_batches):
		start_idx = batch_idx * batch_size
		end_idx = min(start_idx + batch_size, len(reviews))
		batch_reviews = reviews[start_idx:end_idx]

		print(f"批次 {batch_idx + 1}/{total_batches}: {[r['id'] for r in batch_reviews]}")

		batch_results = review_batch(client, model, batch_reviews)

		# 处理结果
		for review, ai_result in zip(batch_reviews, batch_results):
			if ai_result:
				ai_action = ai_result.get("action", "approve")
				ai_names = ai_result.get("verified_names", [])

				result_item = {
					"id": review["id"],
					"original_text": review["original_text"],
					"predicted_names": review["predicted_names"],
					"model_confidence": review["confidence"],
					"ai_verified_names": ai_names,
					"ai_action": ai_action,
					"ai_reason": ai_result.get("reason", ""),
					"human_action": ai_action,
					"human_names": ai_names,
					"human_note": "",
					"created_at": review["created_at"],
					"reviewed_at": datetime.now().isoformat()
				}
				results.append(result_item)
				success_count += 1
				print(f"  ID={review['id']}: {ai_action} -> {ai_names}")
			else:
				print(f"  ID={review['id']}: 失败")

		print()

		# 批次间延迟
		if batch_idx < total_batches - 1:
			time.sleep(0.5)

	output_data = {
		"meta": {
			"total_reviews": len(reviews),
			"success_count": success_count,
			"failed_count": len(reviews) - success_count,
			"model": model,
			"batch_size": batch_size,
			"created_at": datetime.now().isoformat()
		},
		"items": results
	}

	with open(output_path, "w", encoding="utf-8") as f:
		json.dump(output_data, f, ensure_ascii=False, indent=2)

	print(f"完成: {success_count}/{len(reviews)} -> {output_path}")
	print("AI 已填写 human_action/human_names，检查后执行 upload")

	return success_count


def do_upload(input_path: str, dry_run: bool = False) -> int:
	"""
	通过 API 上传人工审核结果。

	Args:
		input_path: 输入 JSON 文件路径
		dry_run: 是否只预览不执行

	Returns:
		成功更新的数量
	"""
	if not os.path.exists(input_path):
		print(f"文件不存在: {input_path}")
		return 0

	with open(input_path, "r", encoding="utf-8") as f:
		data = json.load(f)

	items = data.get("items", [])

	if not items:
		print("没有数据")
		return 0

	to_process = [
		item for item in items
		if item.get("human_action") and item.get("human_action") not in ["skip", "SKIP", ""]
	]

	if not to_process:
		print("没有待上传数据（跳过条目将 human_action 设为 'skip'）")
		return 0

	print(f"待上传: {len(to_process)} 条（跳过 {len(items) - len(to_process)} 条）")

	for item in to_process:
		action = item["human_action"]
		names = item.get("human_names") or item.get("ai_verified_names", [])
		note = item.get("human_note", "")

		print(f"ID={item['id']}: {action} -> {names}")
		print(f"  原文: {item['original_text'][:50]}...")
		if note:
			print(f"  备注: {note}")

	if dry_run:
		print("预览模式，使用 --execute 执行更新")
		return 0

	api_url = get_api_url()
	print(f"API: {api_url}")

	success_count = 0

	for item in to_process:
		try:
			review_id = item["id"]
			action = item["human_action"]
			names = item.get("human_names") or item.get("ai_verified_names", [])
			note = item.get("human_note", "") or ""

			request_body = {
				"action": action,
				"names": names,
				"note": note
			}

			url = f"{api_url}/api/reviews/{review_id}"
			req = urllib.request.Request(
				url,
				data=json.dumps(request_body, ensure_ascii=False).encode("utf-8"),
				headers={"Content-Type": "application/json"},
				method="POST"
			)

			try:
				with urllib.request.urlopen(req, timeout=10) as response:
					result = json.loads(response.read().decode("utf-8"))
					success_count += 1
					print(f"OK ID={review_id}: {action} -> {names}")
			except urllib.error.HTTPError as e:
				error_body = e.read().decode("utf-8") if e.fp else ""
				print(f"FAIL ID={review_id}: HTTP {e.code}")

		except Exception as e:
			print(f"ERROR ID={item['id']}: {e}")

	print(f"完成: {success_count}/{len(to_process)}")

	return success_count


def main():
	parser = argparse.ArgumentParser(
		description="AI 辅助审核姓名识别结果",
		formatter_class=argparse.RawDescriptionHelpFormatter,
		epilog="""
用法:
  python ai_review.py review -c 100 -o result.json   # AI 审核
  python ai_review.py upload -i result.json          # 预览
  python ai_review.py upload -i result.json --execute  # 执行

环境变量: OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL, NER_API_URL
        """
	)

	subparsers = parser.add_subparsers(dest="command", help="子命令")

	# review 子命令
	review_parser = subparsers.add_parser("review", help="执行 AI 审核")
	review_parser.add_argument(
		"--count", "-c",
		type=int,
		default=60,
		help="审核数量 (默认: 60)"
	)
	review_parser.add_argument(
		"--output", "-o",
		type=str,
		default="ai_review_result.json",
		help="输出 JSON 文件路径 (默认: ai_review_result.json)"
	)
	review_parser.add_argument(
		"--model", "-m",
		type=str,
		default=None,
		help="模型名称 (默认: 从环境变量读取)"
	)
	review_parser.add_argument(
		"--batch-size", "-b",
		type=int,
		default=20,
		help="每批处理数量 (默认: 20)"
	)

	# upload 子命令
	upload_parser = subparsers.add_parser("upload", help="上传人工审核结果")
	upload_parser.add_argument(
		"--input", "-i",
		type=str,
		default="training/ai_review_result.json",
		help="输入 JSON 文件路径 (默认: ai_review_result.json)"
	)
	upload_parser.add_argument(
		"--execute",
		action="store_true",
		help="执行实际更新（默认只预览）"
	)

	args = parser.parse_args()

	if args.command == "review":
		do_review(
			count=args.count,
			output_path=args.output,
			model=args.model,
			batch_size=args.batch_size
		)
	elif args.command == "upload":
		do_upload(
			input_path=args.input,
			dry_run=not args.execute
		)
	else:
		parser.print_help()


if __name__ == "__main__":
	main()
