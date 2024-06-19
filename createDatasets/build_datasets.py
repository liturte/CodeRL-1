import os
import json
import pickle

# 定义目录路径
questions_dir = "/data/coding/CodeRL/data/APPS/APPS/test/"
codes_dir = "/data/coding/CodeRL/outputs/python_processe_codes/"
evaluate_dir = "/data/coding/CodeRL/outputs/response_result/"
pkl_dir = "/data/coding/CodeRL/outputs/python_result/"
results_dir = "/data/coding/CodeRL/outputs/test_datasets"

# 确保结果目录存在
os.makedirs(results_dir, exist_ok=True)

# 读取文本文件内容的函数
def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# 读取JSON文件内容的函数
def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

# 读取PKL文件内容的函数
def read_pkl_file(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)

# 准备存储数据集的列表
dataset = []

# 批量处理每个题目和对应的代码
for code_file in os.listdir(codes_dir):
    code_path = os.path.join(codes_dir, code_file)

    if os.path.exists(code_path):
        code_data = read_json_file(code_path)

        for question_number, code_info in code_data.items():
            question_number_padded = question_number.zfill(4)  # 保证题号是四位数
            question_path = os.path.join(questions_dir, question_number_padded, "question.txt")
            solutions_path = os.path.join(questions_dir, question_number_padded, "solutions.json")
            input_output_path = os.path.join(questions_dir, question_number_padded, "input_output.json")
            pkl_file_path = os.path.join(pkl_dir, f"{question_number}.pkl")
            evaluate_file_path = os.path.join(evaluate_dir, f"{question_number_padded}.jsonl")

            # 检查每个文件是否存在
            files_exist = {
                "question_path": os.path.exists(question_path),
                "solutions_path": os.path.exists(solutions_path),
                "input_output_path": os.path.exists(input_output_path),
                "pkl_file_path": os.path.exists(pkl_file_path),
                "evaluate_file_path": os.path.exists(evaluate_file_path)
            }

            # 打印调试信息
            print(f"Processing question {question_number_padded}")
            for file, exists in files_exist.items():
                print(f"{file}: {'exists' if exists else 'does not exist'}")

            if all(files_exist.values()):
                instruction = read_text_file(question_path)
                solutions = read_json_file(solutions_path)
                input_output = read_json_file(input_output_path)
                pkl_data = read_pkl_file(pkl_file_path)
                evaluate_data = read_json_file(evaluate_file_path)

                code_list = code_info["code"]
                generate_code = "\n".join(code_list)

                inputs = input_output.get("inputs", [])
                outputs = input_output.get("outputs", [])
                results = pkl_data.get(int(question_number), {}).get("results", [])
                errors = pkl_data.get(int(question_number), {}).get("errors", [])
                evaluation = evaluate_data.get("evaluation", "")

                # 确保所有对象都可序列化
                def make_serializable(obj):
                    if isinstance(obj, (int, float, str, bool, type(None))):
                        return obj
                    return str(obj)

                # 构建子字段，确保每个结果和错误分别对应一个输入和输出
                io_re_list = []
                for i in range(len(results[0])):
                    if i < len(inputs) and i < len(outputs):
                        io_re_list.append({
                            "input": make_serializable(inputs[i]),
                            "output": make_serializable(outputs[i]),
                            "results": make_serializable(results[0][i]),
                            "error": make_serializable(errors[0][i])
                        })

                # 构建数据条目
                data_entry = {
                    "instruction": make_serializable(instruction),
                    "solutions": make_serializable(solutions),
                    "generate_code": make_serializable(generate_code),
                    "io_results": io_re_list,
                    "evaluate": make_serializable(evaluation)
                }

                dataset.append(data_entry)

# 保存数据集为 JSON Lines 文件
dataset_path = os.path.join(results_dir, "dataset.jsonl")
with open(dataset_path, 'w', encoding='utf-8') as dataset_file:
    for entry in dataset:
        dataset_file.write(json.dumps(entry, ensure_ascii=False) + '\n')

print("数据集构建完成，结果已保存。")
