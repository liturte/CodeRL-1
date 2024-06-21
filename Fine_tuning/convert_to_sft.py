import json

# 定义初始数据集路径和SFT数据集路径
initial_dataset_path = "/data/coding/CodeRL/outputs/test_datasets/cleaned_dataset.jsonl"
sft_dataset_path = "/data/coding/CodeRL/outputs/test_datasets/sft_dataset.jsonl"

# 读取初始数据集并转换为SFT数据集
sft_dataset = []

with open(initial_dataset_path, 'r', encoding='utf-8') as infile:
    for line in infile:
        data_entry = json.loads(line.strip())
        sft_entry = {
            "instruction": data_entry["instruction"].strip(),
            "solution": data_entry["solutions"].strip()
        }
        sft_dataset.append(sft_entry)

# 保存转换后的SFT数据集
with open(sft_dataset_path, 'w', encoding='utf-8') as outfile:
    for entry in sft_dataset:
        json.dump(entry, outfile, ensure_ascii=False)
        outfile.write('\n')

print("数据集转换完成，SFT数据集已保存。")
