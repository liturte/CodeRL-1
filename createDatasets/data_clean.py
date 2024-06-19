import json

# 指定数据集文件路径
dataset_path = "/data/coding/CodeRL/outputs/test_datasets/dataset.jsonl"
cleaned_dataset_path = "/data/coding/CodeRL/outputs/test_datasets/cleaned_dataset.jsonl"

# 读取和清理JSON Lines文件内容
def read_and_clean_jsonl_file(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            entry = json.loads(line)
            # 过滤掉generate_code为空字符串的条目
            if entry.get("generate_code", "").strip():
                data.append(entry)
    return data

# 展示数据集内容
def display_dataset(data, num_entries=5):
    for i, entry in enumerate(data[:num_entries]):
        print(f"Entry {i + 1}:")
        print(json.dumps(entry, indent=4, ensure_ascii=False))
        print("\n" + "="*80 + "\n")

# 保存清理后的数据集为JSON Lines文件
def save_jsonl_file(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        for entry in data:
            file.write(json.dumps(entry, ensure_ascii=False) + '\n')

# 读取并清理数据集
dataset = read_and_clean_jsonl_file(dataset_path)

# 展示前5个条目
display_dataset(dataset, num_entries=5)

# 保存清理后的数据集
save_jsonl_file(dataset, cleaned_dataset_path)

print("数据集读取、清理和保存完成。")
