import os
import json
import openai

# 设置你的OpenAI API密钥
openai.api_key = 'sk-3rVkTm4NdFx3sBEwK6o7g9OWAnXR5OUC7w3OuaZObyaR9oJZ'
openai.api_base = "https://api.chatanywhere.tech/v1"

# 题目和代码的目录路径
questions_dir = "/data/coding/CodeRL/data/APPS/APPS/test/"
codes_dir = "/data/coding/CodeRL/outputs/python_processe_codes/"
results_dir = "/data/coding/CodeRL/outputs/response_result"  # 保存评估结果的路径

# 确保结果目录存在
os.makedirs(results_dir, exist_ok=True)


# 读取文本文件内容的函数
def read_text_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None


# 读取JSON文件内容的函数
def read_json_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except Exception as e:
        print(f"Error reading JSON file {file_path}: {e}")
        return None


# 使用ChatGPT API对代码进行评估的函数
def evaluate_code(question, code):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a code evaluation assistant."},
                {"role": "user", "content": f"Question: {question}\nCode:\n{code}\nPlease evaluate this code."}
            ],
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        print(f"Error during API call: {e}")
        return "Error during API call"


# 批量处理每个题目和对应的代码
for code_file in os.listdir(codes_dir):
    code_path = os.path.join(codes_dir, code_file)

    if os.path.exists(code_path):
        code_data = read_json_file(code_path)
        if code_data is None:
            continue

        for question_number, code_info in code_data.items():
            code_list = code_info["code"]
            code = "\n".join(code_list)
            question_number_padded = question_number.zfill(4)  # 保证题号是四位数
            question_path = os.path.join(questions_dir, question_number_padded, "question.txt")

            if os.path.exists(question_path):
                question = read_text_file(question_path)
                if question is None:
                    continue

                evaluation = evaluate_code(question, code)
                result_file_path = os.path.join(results_dir, f"{question_number_padded}.jsonl")

                try:
                    with open(result_file_path, 'a', encoding='utf-8') as result_file:
                        json.dump({
                            "question": question,
                            "code": code,
                            "evaluation": evaluation
                        }, result_file, ensure_ascii=False)
                        result_file.write('\n')
                        print(f"Evaluation for question {question_number_padded} saved.")
                except Exception as e:
                    print(f"Error writing to result file {result_file_path}: {e}")

print("评估完成，结果已保存。")
