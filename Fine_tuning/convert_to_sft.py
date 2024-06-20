import json


def convert_to_sft_format(data_entry):
    instruction = data_entry.get("instruction", "")
    generate_code = data_entry.get("generate_code", "")
    solutions = data_entry.get("solutions", "")

    context = f"Instruction: {instruction}\nCorrect Solution: {solutions}\nGenerated Code: {generate_code}\nEvaluation: {data_entry.get('evaluate', '')}\n\n"

    qa_pairs = []
    for io_result in data_entry.get("io_results", []):
        input_value = io_result.get("input", "")
        output_value = io_result.get("output", "")
        result = io_result.get("result", None)

        question = f"Given input: {input_value}, expected output: {output_value}. What is the test result?"
        answer = f"Result: {'Pass' if result else 'Fail'}"

        qa_pairs.append({"question": question, "answer": answer})

    return {
        "context": context,
        "qa_pairs": qa_pairs
    }


# 读取原始数据集
dataset_path = "/data/coding/CodeRL/outputs/test_datasets/cleaned_dataset.jsonl"
sft_dataset = []

with open(dataset_path, 'r', encoding='utf-8') as file:
    for line in file:
        data_entry = json.loads(line.strip())
        sft_format_entry = convert_to_sft_format(data_entry)
        sft_dataset.append(sft_format_entry)

# 保存转换后的数据集
sft_dataset_path = "/data/coding/CodeRL/outputs/test_datasets/sft_dataset.jsonl"
with open(sft_dataset_path, 'w', encoding='utf-8') as file:
    for entry in sft_dataset:
        file.write(json.dumps(entry, ensure_ascii=False) + '\n')

print("数据集转换完成，结果已保存。")
