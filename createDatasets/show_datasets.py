import json

# 指定数据集文件路径
dataset_path = "/data/coding/CodeRL/outputs/test_datasets/cleaned_dataset.jsonl"

def read_jsonl_file(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data
def display_dataset(data, num_entries=5):
    for i, entry in enumerate(data[:num_entries]):
        print(f"Entry {i + 1}:")
        print(json.dumps(entry, indent=4, ensure_ascii=False))
        print("\n" + "="*80 + "\n")

dataset = read_jsonl_file(dataset_path)

display_dataset(dataset, num_entries=5)
