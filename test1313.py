import json

# 读取 JSON 文件
with open('C:/Users/Administrator/Desktop/新闻稿ppt照片/1043.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# 提取 answer 后的内容
modified_data = {}
for key, value in data.items():
    if 'code' in value:
        code = value['code']
        if isinstance(code, list) and code:
            # 找到 "ANSWER:" 的位置并提取之后的内容
            code_str = code[0]
            answer_index = code_str.find("ANSWER:")
            if answer_index != -1:
                answer_content = code_str[answer_index + len("ANSWER:"):].strip()
                modified_data[key] = answer_content

# 将修改后的数据写回到新的 JSON 文件
with open('C:/Users/Administrator/Desktop/新闻稿ppt照片/new_1043.json', 'w', encoding='utf-8') as file:
    json.dump(modified_data, file, ensure_ascii=False, indent=4)

print("已成功保存修改后的内容到 modified_1043.json")
