#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import os
import random

CHINESE_NAMES_CORPUS_PATH = "../data/Chinese_Names_Corpus.txt"
RAW_OUTPUT_PATH = "../data/raw_name_corpus.jsonl"
BIO_OUTPUT_PATH = "../data/bio_name_corpus.jsonl"

CS_PREFIXES = [
	"您好，", "您好!", "请问，", "麻烦问一下，", "不好意思，",
	"您好请问", "您好想咨询一下", "您好有个问题", "打扰了，",
	"客服为您服务，", "您好，这边是客服，", "您好我是客服，",
]

CS_SUFFIXES = [
	"。", "，谢谢。", "，麻烦您了。", "，辛苦了。", "，拜托了。",
	"，麻烦帮我处理一下。", "，请核实一下。", "，谢谢您的帮助。",
	"，能帮我看看吗？", "，麻烦尽快处理。", "，谢谢配合。",
]

BIZ_PREFIXES = [
	"关于这个项目，", "针对这个方案，", "关于上次讨论的，",
	"根据安排，", "按照计划，", "经确认，",
	"需要说明的是，", "根据反馈，", "关于这个议题，",
]

BIZ_SUFFIXES = [
	"，请知悉。", "，请查阅。", "，请确认。", "，请跟进。",
	"，烦请处理。", "，谢谢配合。", "，请尽快落实。",
]

SINGLE_PATTERNS = [
	"{name}的订单状态需要确认一下",
	"{name}的包裹显示已签收",
	"{name}的快递正在派送中",
	"{name}的订单地址需要修改",
	"{name}的包裹在转运中心",
	"{name}的订单已经发货了",
	"{name}的快递已经送达",
	"{name}的会员卡即将到期",
	"{name}的账户需要激活",
	"{name}的积分快过期了",
	"{name}的账号异常",
	"{name}的密码需要重置",
	"{name}的预约已确认",
	"{name}的预约需要取消",
	"{name}的服务已经完成",
	"{name}的预约时间已调整",
	"{name}反馈的问题已处理",
	"{name}咨询的事项正在跟进",
	"{name}反映的情况属实",
	"{name}提出的建议已采纳",
	"{name}的投诉已受理",
	"{name}的问题正在排查",
	"{name}的退款已到账",
	"{name}的维修已完成",
	"请通知{name}参加会议",
	"麻烦转告{name}一下",
	"请{name}尽快回复",
	"让{name}回个电话",
	"{name}在找您",
	"{name}刚才来过电话",
	"{name}让您回电",
	"这是{name}的报表",
	"这份文件{name}已经审核过了",
	"{name}的报告需要签字",
	"{name}的申请已经提交",
	"{name}的材料需要补充",
	"您好我是{name}",
	"我是{name}，想咨询一下",
	"你好，我叫{name}",
	"您好，本人{name}",
	"请问{name}在吗",
	"{name}有空吗",
	"{name}方便接电话吗",
	"{name}现在能来一下吗",
	"谁知道{name}的联系方式",
	"{name}的工位在哪里",
	"{name}在会议室",
	"{name}去客户那里了",
	"{name}出差了",
	"{name}今天请假",
]

DOUBLE_PATTERNS = [
	"{name1}和{name2}负责这个项目",
	"{name1}、{name2}在等您",
	"{name1}与{name2}共同负责",
	"{name1}、{name2}已经到了",
	"{name1}和{name2}约好了见面",
	"{name1}的订单和{name2}的订单弄混了",
	"{name1}的报告{name2}还没看",
	"{name1}说{name2}知道详情",
	"{name1}让{name2}转告您",
	"{name1}的意见是{name2}说的对",
	"请{name1}把资料给{name2}",
	"{name1}的事情转给{name2}处理",
	"{name1}的工单{name2}在跟进",
	"{name1}的工作{name2}接手了",
	"请通知{name1}和{name2}开会",
	"{name1}和{name2}都需要到场",
	"麻烦{name1}叫{name2}过来",
	"{name1}和{name2}谁有空",
	"{name1}、{name2}在哪个部门",
	"{name1}和{name2}什么关系",
	"{name1}说{name2}也会来",
	"{name1}知道{name2}的联系方式吗",
	"这是{name1}和{name2}的合影",
	"{name1}推荐了{name2}的产品",
]

TRIPLE_PATTERNS = [
	"{name1}、{name2}、{name3}都在会议室",
	"请通知{name1}、{name2}和{name3}开会",
	"{name1}、{name2}、{name3}约好了见面",
	"{name1}、{name2}和{name3}都在等您",
	"这个项目{name1}、{name2}、{name3}共同负责",
	"{name1}、{name2}和{name3}是一个团队的",
	"{name1}、{name2}、{name3}的申请已通过",
	"这项工作由{name1}、{name2}、{name3}协作完成",
	"{name1}说{name2}和{name3}知道情况",
	"请{name1}转告{name2}和{name3}",
	"{name1}让{name2}通知{name3}",
	"{name1}、{name2}、{name3}三个人的订单都到了",
	"{name1}比{name2}和{name3}来得早",
	"{name1}、{name2}、{name3}谁的工单还没处理",
	"这是{name1}、{name2}和{name3}的合影",
	"{name1}、{name2}、{name3}都在这个群里",
]

NO_NAME_SENTENCES = [
	"您的包裹平安送达了",
	"问题已经解决了",
	"来自北京的订单",
	"上海发货的快递",
	"您好，这里是服务中心",
	"系统显示您的订单已完成",
	"感谢您选择我们的服务",
	"这个方案真的很棒",
	"安全问题很重要",
	"中华人民共和国成立了",
	"新时代的机遇",
	"东方明珠塔",
	"产品质量有保障",
	"服务态度非常好",
	"物流速度很快",
	"价格很实惠",
	"包装很精美",
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
	"您的订单号是202401010001",
	"预约时间是下周三下午",
	"快递单号请查收",
	"订单三天内送达",
	"服务时间为早九晚六",
	"故宫博物院",
	"长城旅游",
	"黄河入海流",
	"长江大桥",
	"泰山日出",
	"西湖美景",
	"黄山迎客松",
]

EDGE_CASE_PATTERNS = [
	"{name}，您好",
	"{name}！在吗",
	"{name}～有空吗",
	"{name}，请稍等",
	"{name}，有您的快递",
	"{name}，电话找您",
	"您好，请问您是{name}吗",
	"您找的人是{name}",
	"来电显示是{name}",
	"快递收件人是{name}",
	"这个包裹是{name}的",
	"刚才来访的是{name}",
	"关于{name}的事，是这样",
	"之前{name}、还有其他人也提过",
	"就是{name}，对",
	"那个{name}——他刚走",
	"请问是{name}本人吗",
	"找一下{name}先生",
	"麻烦叫一下{name}女士",
	"请问{name}工位在哪",
	"请问{name}在哪个部门",
	"这个订单的收件人是{name}",
]


def load_names_from_file(file_path: str, encoding: str = "utf-8") -> list:
	names = []
	if not os.path.exists(file_path):
		return names

	with open(file_path, "r", encoding=encoding) as f:
		for line in f:
			line = line.strip()
			if not line or line.startswith("#"):
				continue

			if "|" in line:
				name = line.split("|")[0].strip()
			else:
				name = line.strip()

			if name and len(name) >= 2 and len(name) <= 10:
				names.append(name)

	return names


def build_sentence(pattern: str, names: list, prefix: str = "", suffix: str = "") -> tuple:
	if len(names) == 1:
		text = pattern.format(name=names[0])
	elif len(names) == 2:
		text = pattern.format(name1=names[0], name2=names[1])
	elif len(names) == 3:
		text = pattern.format(name1=names[0], name2=names[1], name3=names[2])
	else:
		text = pattern

	return prefix + text + suffix, names


def generate_edge_case(chinese_names: list) -> dict:
	pattern = random.choice(EDGE_CASE_PATTERNS)
	name = random.choice(chinese_names)

	text = pattern.replace("{name}", name)
	return {"text": text, "entities": [name]}


def generate_corpus(chinese_names: list, output_file: str, total_count: int = 50000):
	single_count = int(total_count * 0.45)
	double_count = int(total_count * 0.25)
	triple_count = int(total_count * 0.05)
	edge_count = int(total_count * 0.10)
	no_name_count = int(total_count * 0.15)

	print(f"生成计划:")
	print(f"  - 单人中文姓名: {single_count}")
	print(f"  - 双人中文姓名: {double_count}")
	print(f"  - 三人中文姓名: {triple_count}")
	print(f"  - 边界情况: {edge_count}")
	print(f"  - 无姓名干扰: {no_name_count}")
	print(f"  - 总计: {total_count}")
	print("-" * 50)

	generated = []

	for _ in range(single_count):
		pattern = random.choice(SINGLE_PATTERNS)
		name = random.choice(chinese_names)
		prefix = random.choice(CS_PREFIXES) if random.random() < 0.3 else ""
		suffix = random.choice(CS_SUFFIXES) if random.random() < 0.3 else ""
		text, entities = build_sentence(pattern, [name], prefix, suffix)
		generated.append({"text": text, "entities": entities})

	for _ in range(double_count):
		pattern = random.choice(DOUBLE_PATTERNS)
		names = random.sample(chinese_names, 2)
		prefix = random.choice(BIZ_PREFIXES) if random.random() < 0.4 else ""
		suffix = random.choice(BIZ_SUFFIXES) if random.random() < 0.4 else ""
		text, entities = build_sentence(pattern, names, prefix, suffix)
		generated.append({"text": text, "entities": entities})

	for _ in range(triple_count):
		pattern = random.choice(TRIPLE_PATTERNS)
		names = random.sample(chinese_names, 3)
		text, entities = build_sentence(pattern, names)
		generated.append({"text": text, "entities": entities})

	for _ in range(edge_count):
		item = generate_edge_case(chinese_names)
		generated.append(item)

	for _ in range(no_name_count):
		text = random.choice(NO_NAME_SENTENCES)
		generated.append({"text": text, "entities": []})

	random.shuffle(generated)

	with open(output_file, "w", encoding="utf-8") as f:
		for item in generated:
			f.write(json.dumps(item, ensure_ascii=False) + "\n")

	print(f"已生成 {len(generated)} 条语料，保存到: {output_file}")
	return len(generated)


def create_sample_names():
	surnames = [
		"王", "李", "张", "刘", "陈", "杨", "黄", "赵", "周", "吴",
		"徐", "孙", "马", "朱", "胡", "郭", "何", "高", "林", "罗",
		"郑", "梁", "谢", "宋", "唐", "许", "韩", "冯", "邓", "曹",
		"彭", "曾", "萧", "田", "董", "袁", "潘", "于", "蒋", "蔡",
		"余", "杜", "叶", "程", "苏", "魏", "吕", "丁", "任", "沈",
		"欧阳", "司马", "诸葛", "上官", "皇甫", "东方", "令狐", "慕容", "轩辕", "公孙",
	]

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

	return chinese_names


def main():
	print("=" * 60)
	print("姓名语料生成器 v2.2")
	print("=" * 60)

	print("\n[1] 准备姓名数据...")
	chinese_names = create_sample_names()

	if os.path.exists(CHINESE_NAMES_CORPUS_PATH):
		names = load_names_from_file(CHINESE_NAMES_CORPUS_PATH)
		chinese_names.extend(names)
		print(f"  从 {CHINESE_NAMES_CORPUS_PATH} 加载中文姓名: +{len(names)}")

	chinese_names = list(set(chinese_names))
	print(f"  去重后中文姓名: {len(chinese_names)} 个")

	print("\n[2] 生成训练语料...")
	generate_corpus(
		chinese_names=chinese_names,
		output_file=RAW_OUTPUT_PATH,
		total_count=50000,
	)

	print("\n[3] 转换为 BIO 格式...")
	from generate_bio_tags import process_ai_generated_data
	process_ai_generated_data(RAW_OUTPUT_PATH, BIO_OUTPUT_PATH)

	print("\n" + "=" * 60)
	print("完成！")
	print("=" * 60)


if __name__ == "__main__":
	main()
