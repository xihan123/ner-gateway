#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
姓名语料生成器 v2.0 - 更合理的句子模板设计

数据来源：
1. CCNC: 365万+ 中文姓名 (https://github.com/jaaack-wang/ccnc)
2. Chinese-Names-Corpus: 120万中文人名 + 48万翻译人名 (https://github.com/wainshine/Chinese-Names-Corpus)

输出格式：
    JSONL 文件，格式：{"text": "生成的句子", "entities": ["姓名1", "姓名2"]}

优化点：
1. 模板组件化：前缀 + 主体 + 后缀，动态组合
2. 多场景覆盖：客服、商务、日常、内部沟通
3. 边界情况：姓名位置、尊称剥离、干扰项
4. 真实感：模拟真实对话场景
"""

import json
import os
import random

# ============================================================
# 句子组件 - 动态组合生成更自然的句子
# ============================================================

# 客服场景 - 前缀（开场白）
CS_PREFIXES = [
	"您好，", "您好!", "请问，", "麻烦问一下，", "不好意思，",
	"您好请问", "您好想咨询一下", "您好有个问题", "打扰了，",
	"客服为您服务，", "您好，这边是客服，", "您好我是客服，",
]

# 客服场景 - 后缀（结束语）
CS_SUFFIXES = [
	"。", "，谢谢。", "，麻烦您了。", "，辛苦了。", "，拜托了。",
	"，麻烦帮我处理一下。", "，请核实一下。", "，谢谢您的帮助。",
	"，能帮我看看吗？", "，麻烦尽快处理。", "，谢谢配合。",
]

# 商务场景 - 前缀
BIZ_PREFIXES = [
	"关于这个项目，", "针对这个方案，", "关于上次讨论的，",
	"根据安排，", "按照计划，", "经确认，",
	"需要说明的是，", "根据反馈，", "关于这个议题，",
]

# 商务场景 - 后缀
BIZ_SUFFIXES = [
	"，请知悉。", "，请查阅。", "，请确认。", "，请跟进。",
	"，烦请处理。", "，谢谢配合。", "，请尽快落实。",
]

# 主体句型模板 - 使用 {name} 占位符
# 单人场景
SINGLE_PATTERNS = [
	# 订单/包裹相关
	"{name}的订单状态需要确认一下",
	"{name}的包裹显示已签收",
	"{name}的快递正在派送中",
	"{name}的订单地址需要修改",
	"{name}的包裹在转运中心",
	"{name}的订单已经发货了",
	"{name}的快递已经送达",

	# 账户/会员相关
	"{name}的会员卡即将到期",
	"{name}的账户需要激活",
	"{name}的积分快过期了",
	"{name}的账号异常",
	"{name}的密码需要重置",

	# 预约/服务相关
	"{name}的预约已确认",
	"{name}的预约需要取消",
	"{name}的服务已经完成",
	"{name}的预约时间已调整",

	# 咨询/反馈相关
	"{name}反馈的问题已处理",
	"{name}咨询的事项正在跟进",
	"{name}反映的情况属实",
	"{name}提出的建议已采纳",

	# 投诉/问题相关
	"{name}的投诉已受理",
	"{name}的问题正在排查",
	"{name}的退款已到账",
	"{name}的维修已完成",

	# 通知/提醒相关
	"请通知{name}参加会议",
	"麻烦转告{name}一下",
	"请{name}尽快回复",
	"让{name}回个电话",
	"{name}在找您",
	"{name}刚才来过电话",
	"{name}让您回电",

	# 文件/工作相关
	"这是{name}的报表",
	"这份文件{name}已经审核过了",
	"{name}的报告需要签字",
	"{name}的申请已经提交",
	"{name}的材料需要补充",

	# 自我介绍
	"您好我是{name}",
	"我是{name}，想咨询一下",
	"你好，我叫{name}",
	"您好，本人{name}",

	# 询问类
	"请问{name}在吗",
	"{name}有空吗",
	"{name}方便接电话吗",
	"{name}现在能来一下吗",
	"谁知道{name}的联系方式",

	# 位置/地点相关
	"{name}的工位在哪里",
	"{name}在会议室",
	"{name}去客户那里了",
	"{name}出差了",
	"{name}今天请假",
]

# 双人场景 - 使用 {name1} 和 {name2} 占位符
DOUBLE_PATTERNS = [
	# 协作类
	"{name1}和{name2}负责这个项目",
	"{name1}、{name2}在等您",
	"{name1}与{name2}共同负责",
	"{name1}、{name2}已经到了",
	"{name1}和{name2}约好了见面",

	# 对比类
	"{name1}的订单和{name2}的订单弄混了",
	"{name1}的报告{name2}还没看",
	"{name1}说{name2}知道详情",
	"{name1}让{name2}转告您",
	"{name1}的意见是{name2}说的对",

	# 转交类
	"请{name1}把资料给{name2}",
	"{name1}的事情转给{name2}处理",
	"{name1}的工单{name2}在跟进",
	"{name1}的工作{name2}接手了",

	# 通知类
	"请通知{name1}和{name2}开会",
	"{name1}和{name2}都需要到场",
	"麻烦{name1}叫{name2}过来",

	# 询问类
	"{name1}和{name2}谁有空",
	"{name1}、{name2}在哪个部门",
	"{name1}和{name2}什么关系",

	# 组合类
	"{name1}说{name2}也会来",
	"{name1}知道{name2}的联系方式吗",
	"这是{name1}和{name2}的合影",
	"{name1}推荐了{name2}的产品",
]

# 三人场景
TRIPLE_PATTERNS = [
	"{name1}、{name2}、{name3}都在会议室",
	"这个项目{name1}、{name2}、{name3}共同负责",
	"请通知{name1}、{name2}和{name3}",
	"{name1}、{name2}、{name3}的申请已通过",
	"{name1}说{name2}和{name3}知道情况",
]

# 尊称后缀 - 这些需要从实体中剥离
TITLE_SUFFIXES = [
	"总", "经理", "总监", "主管", "主任", "科长", "部长",
	"老师", "医生", "护士", "教授", "律师", "会计",
	"哥", "姐", "弟", "妹", "叔", "婶", "爷", "奶",
	"先生", "女士", "小姐", "夫人",
	"工", "师傅",  # 工程师、技师
]

# 干扰句 - 无姓名但容易误判
NO_NAME_SENTENCES = [
	# 地名/机构名
	"您的包裹平安送达了",
	"问题已经平安解决了",
	"来自北京的订单",
	"上海发货的快递",
	"您好，这里是服务中心",
	"系统显示您的订单已完成",
	"感谢您选择我们的服务",

	# 容易误判为姓名的词
	"这个方案真的很棒",
	"安全问题很重要",
	"张三丰太极拳很出名",
	"中华人民共和国成立了",
	"平安保险公司",
	"李宁运动品牌",
	"王老吉凉茶",
	"老北京炸酱面",
	"新时代的机遇",
	"东方明珠塔",

	# 纯客服对话
	"您好请问有什么可以帮您",
	"请问您需要什么服务",
	"感谢您的耐心等待",
	"您的意见我们已记录",
	"请问还有其他问题吗",
	"祝您生活愉快",
	"感谢您的支持",
	"请稍等正在查询",
	"好的为您处理",
	"已为您办理成功",

	# 数字/日期类
	"您的订单号是202401010001",
	"预约时间是下周三下午",
	"快递单号请查收",
	"订单三天内送达",
	"服务时间为早九晚六",
]

# 边界情况模板 - 姓名位置特殊
EDGE_CASE_PATTERNS = [
	# 姓名在句首
	"{name}，您好",
	"{name}！在吗",
	"{name}～有空吗",

	# 姓名在句尾
	"您好，请问您是{name}吗",
	"您找的人是{name}",
	"来电显示是{name}",
	"快递收件人是{name}",

	# 姓名被标点分隔
	"关于{name}的事，是这样",
	"之前{name}、还有其他人也提过",

	# 英文姓名特殊格式
	"Mr. {name} left a message",
	"Hello, this is {name} speaking",
	"请问{name}在吗",
]


def load_names_from_file(file_path: str, encoding: str = "utf-8") -> list:
	"""从文件加载姓名列表"""
	names = []
	if not os.path.exists(file_path):
		return names

	with open(file_path, "r", encoding=encoding) as f:
		for line in f:
			line = line.strip()
			if not line or line.startswith("#"):
				continue

			# 处理不同格式
			if "|" in line:
				# 格式：中文|英文|性别，取中文部分
				name = line.split("|")[0].strip()
			else:
				name = line.strip()

			# 过滤无效姓名
			if name and len(name) >= 2 and len(name) <= 10:
				names.append(name)

	return names


def build_sentence(pattern: str, names: list, prefix: str = "", suffix: str = "") -> tuple:
	"""
	构建句子

	Args:
		pattern: 句型模板
		names: 姓名列表
		prefix: 前缀
		suffix: 后缀

	Returns:
		(句子, 实体列表)
	"""
	if len(names) == 1:
		text = pattern.format(name=names[0])
	elif len(names) == 2:
		text = pattern.format(name1=names[0], name2=names[1])
	elif len(names) == 3:
		text = pattern.format(name1=names[0], name2=names[1], name3=names[2])
	else:
		text = pattern

	return prefix + text + suffix, names


def generate_with_title(names: list, patterns: list) -> dict:
	"""生成带尊称的句子（需要剥离尊称）"""
	name = random.choice(names)
	title = random.choice(TITLE_SUFFIXES)
	pattern = random.choice(patterns)

	# 原始姓名 + 尊称
	full_call = name + title

	# 替换模板中的 {name} 为带尊称的称呼
	text = pattern.replace("{name}", full_call)

	# 实体仍然是原始姓名（尊称需要剥离）
	# 但如果是复姓+单字姓+尊称的情况，可能需要特殊处理
	# 这里简化处理：实体是去掉尊称后的部分
	return {"text": text, "entities": [name]}


def generate_edge_case(names: list, chinese_names: list, english_names: list) -> dict:
	"""生成边界情况句子"""
	pattern = random.choice(EDGE_CASE_PATTERNS)

	# 根据模板决定使用中文还是英文名
	if "Mr." in pattern or "Hello" in pattern or "this is" in pattern:
		name = random.choice(english_names) if english_names else random.choice(names)
	else:
		name = random.choice(chinese_names) if chinese_names else random.choice(names)

	text = pattern.replace("{name}", name)
	return {"text": text, "entities": [name]}


def generate_corpus(
		chinese_names: list,
		english_names: list,
		output_file: str,
		total_count: int = 50000,
):
	"""
	生成训练语料

	数据分布：
	- 45% 单人中文姓名
	- 20% 双人中文姓名
	- 5% 三人中文姓名
	- 10% 英文姓名
	- 10% 带尊称（需剥离）
	- 5% 边界情况
	- 5% 无姓名干扰
	"""
	all_names = chinese_names + english_names

	# 计算各类数量
	single_count = int(total_count * 0.45)
	double_count = int(total_count * 0.20)
	triple_count = int(total_count * 0.05)
	english_count = int(total_count * 0.10)
	title_count = int(total_count * 0.10)
	edge_count = int(total_count * 0.05)
	no_name_count = int(total_count * 0.05)

	print(f"生成计划:")
	print(f"  - 单人中文姓名: {single_count}")
	print(f"  - 双人中文姓名: {double_count}")
	print(f"  - 三人中文姓名: {triple_count}")
	print(f"  - 英文姓名: {english_count}")
	print(f"  - 带尊称(需剥离): {title_count}")
	print(f"  - 边界情况: {edge_count}")
	print(f"  - 无姓名干扰: {no_name_count}")
	print(f"  - 总计: {total_count}")
	print("-" * 50)

	generated = []

	# 1. 生成单人中文姓名句子
	for _ in range(single_count):
		pattern = random.choice(SINGLE_PATTERNS)
		name = random.choice(chinese_names)

		# 随机添加前缀后缀
		prefix = random.choice(CS_PREFIXES) if random.random() < 0.3 else ""
		suffix = random.choice(CS_SUFFIXES) if random.random() < 0.3 else ""

		text, entities = build_sentence(pattern, [name], prefix, suffix)
		generated.append({"text": text, "entities": entities})

	# 2. 生成双人中文姓名句子
	for _ in range(double_count):
		pattern = random.choice(DOUBLE_PATTERNS)
		names = random.sample(chinese_names, 2)

		prefix = random.choice(BIZ_PREFIXES) if random.random() < 0.4 else ""
		suffix = random.choice(BIZ_SUFFIXES) if random.random() < 0.4 else ""

		text, entities = build_sentence(pattern, names, prefix, suffix)
		generated.append({"text": text, "entities": entities})

	# 3. 生成三人中文姓名句子
	for _ in range(triple_count):
		pattern = random.choice(TRIPLE_PATTERNS)
		names = random.sample(chinese_names, 3)

		text, entities = build_sentence(pattern, names)
		generated.append({"text": text, "entities": entities})

	# 4. 生成英文姓名句子
	for _ in range(english_count):
		name = random.choice(english_names)

		# 英文名使用部分中文模板
		patterns = SINGLE_PATTERNS[:10] + EDGE_CASE_PATTERNS[-3:]
		pattern = random.choice(patterns)

		text = pattern.replace("{name}", name)
		generated.append({"text": text, "entities": [name]})

	# 5. 生成带尊称的句子（需要剥离尊称）
	for _ in range(title_count):
		item = generate_with_title(chinese_names, SINGLE_PATTERNS[:15])
		generated.append(item)

	# 6. 生成边界情况
	for _ in range(edge_count):
		item = generate_edge_case(all_names, chinese_names, english_names)
		generated.append(item)

	# 7. 生成无姓名干扰句
	for _ in range(no_name_count):
		text = random.choice(NO_NAME_SENTENCES)
		generated.append({"text": text, "entities": []})

	# 打乱顺序
	random.shuffle(generated)

	# 保存文件
	with open(output_file, "w", encoding="utf-8") as f:
		for item in generated:
			f.write(json.dumps(item, ensure_ascii=False) + "\n")

	print(f"已生成 {len(generated)} 条语料，保存到: {output_file}")
	return len(generated)


def create_sample_names():
	"""创建示例姓名数据"""
	# 常见中文姓氏
	surnames = [
		"王", "李", "张", "刘", "陈", "杨", "黄", "赵", "周", "吴",
		"徐", "孙", "马", "朱", "胡", "郭", "何", "高", "林", "罗",
		"郑", "梁", "谢", "宋", "唐", "许", "韩", "冯", "邓", "曹",
		"彭", "曾", "萧", "田", "董", "袁", "潘", "于", "蒋", "蔡",
		"余", "杜", "叶", "程", "苏", "魏", "吕", "丁", "任", "沈",
		"欧阳", "司马", "诸葛", "上官", "皇甫", "东方", "令狐", "慕容", "轩辕", "公孙",
	]

	# 常见名字
	given_names = [
		"伟", "芳", "娜", "敏", "静", "丽", "强", "磊", "军", "洋",
		"勇", "艳", "杰", "娟", "涛", "明", "超", "秀", "霞", "平",
		"建国", "建军", "志强", "志明", "海燕", "小红", "小明", "小华", "子轩", "子涵",
		"雨泽", "宇轩", "浩宇", "欣怡", "诗涵", "梦琪", "雨桐", "梓涵", "佳琪", "一诺",
	]

	chinese_names = []
	for surname in surnames:
		for given in given_names:
			name = surname + given
			if len(name) >= 2:
				chinese_names.append(name)

	english_names = [
		"John", "Mary", "David", "Sarah", "Michael", "Lisa", "James", "Emma",
		"Tom", "Jerry", "Jack", "Rose", "Lucy", "Lily", "Tony", "Mike", "Alice",
		"Bob", "Kevin", "Nancy", "Sam", "Tina", "Victor", "Wendy", "Zack", "Zoe",
	]

	return chinese_names, english_names


def main():
	"""主函数"""
	print("=" * 60)
	print("姓名语料生成器 v2.0")
	print("=" * 60)

	# 准备姓名数据
	print("\n[1] 准备姓名数据...")
	chinese_names, english_names = create_sample_names()

	# 加载外部文件
	for file_path, name_attr in [
		("Chinese_Names_Corpus.txt", "中文姓名"),
		("English_Cn_Name_Corpus.txt", "英文姓名"),
	]:
		if os.path.exists(file_path):
			names = load_names_from_file(file_path)
			if "中文" in name_attr:
				chinese_names.extend(names)
			else:
				english_names.extend(names)
			print(f"  从 {file_path} 加载{name_attr}: +{len(names)}")

	# 去重
	chinese_names = list(set(chinese_names))
	english_names = list(set(english_names))
	print(f"  去重后中文姓名: {len(chinese_names)} 个")
	print(f"  去重后英文姓名: {len(english_names)} 个")

	# 生成语料
	print("\n[2] 生成训练语料...")
	output_file = "raw_name_corpus.jsonl"
	generate_corpus(
		chinese_names=chinese_names,
		english_names=english_names,
		output_file=output_file,
		total_count=50000,
	)

	# 转换为 BIO 格式
	print("\n[3] 转换为 BIO 格式...")
	from generate_bio_tags import process_ai_generated_data
	process_ai_generated_data(output_file, "bio_name_corpus.jsonl")

	print("\n" + "=" * 60)
	print("完成！")
	print("=" * 60)


if __name__ == "__main__":
	main()
