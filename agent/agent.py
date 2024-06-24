import json
import openai
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
from tqdm import tqdm

# 读取HumanEval数据集
def read_humaneval(file_path):
    tasks = []
    with open(file_path, 'r') as f:
        for line in f:
            tasks.append(json.loads(line))
    return tasks

# 提示函数
def standard_prompt(prompt: str) -> str:
    return f"""Complete the following Python code without any tests or explanation,Do not output irrelevant commit\n{prompt}"""

# 生成代码
def generate_codes(model, tokenizer, prompt, num_samples=1, max_length=2048):
    inputs = tokenizer(prompt, return_tensors='pt').to('cuda')
    codes = []
    for _ in range(num_samples):
        outputs = model.generate(inputs['input_ids'], max_length=max_length)
        code = tokenizer.decode(outputs[0], skip_special_tokens=True)
        codes.append(code)
    return codes

# 使用ChatGPT评估生成的代码
def evaluate_code_with_chatgpt(prompt, code):
    messages = [
        {"role": "system", "content": "You are an expert code reviewer."},
        {"role": "user", "content": f"Here is a coding problem and its solution. Please review the code for correctness.\nProblem: {prompt}\nSolution: {code}"}
    ]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    return response['choices'][0]['message']['content']

# 保存生成的代码为JSON格式
def save_generated_code(file_path, result):
    with open(file_path, 'a') as f:
        f.write(json.dumps(result) + '\n')

def main():
    # 设置OpenAI API密钥
    openai.api_key = 'sk-8XGqiRc9787Q19pnlyrGu5L6Gm7z81fIv2Rv1JWlmmCs6GkY'
    openai.api_base = "https://api.chatanywhere.tech/v1"

    # 初始化模型和tokenizer
    model_name = "/data/coding/CodeUltraFeedback/models"
    model = AutoModelForCausalLM.from_pretrained(model_name).to('cuda')
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 读取数据集
    humaneval_path = "/data/coding/CodeUltraFeedback/code-eval/human-eval/data/HumanEval.jsonl"
    tasks = read_humaneval(humaneval_path)

    # 设置从哪个任务开始处理
    start_task_id = "HumanEval/64"
    start_index = next((i for i, task in enumerate(tasks) if task['task_id'] == start_task_id), 0)
    tasks = tasks[start_index:]

    results = []

    # 生成代码并保存
    for task in tqdm(tasks):
        task_id = task['task_id']
        prompt = task['prompt']
        formatted_prompt = standard_prompt(prompt)  # 使用标准提示

        # Step 1: 使用模型生成初始代码
        generated_codes = generate_codes(model, tokenizer, formatted_prompt, num_samples=1)

        for i, generated_code in enumerate(generated_codes):
            # 打印生成的代码
            print(f"Generated code for task {task_id} (initial):")
            print(generated_code)
            print("\n")
f
            # Step 2: 使用ChatGPT评估生成的代码
            try:
                feedback = evaluate_code_with_chatgpt(prompt, generated_code)
                print(f"Feedback for task {task_id}:")
                print(feedback)
                print("\n")
            except Exception as e:
                print(f"Error evaluating code for task {task_id}: {e}")
                feedback = "Error evaluating code."

            # Step 3: 根据ChatGPT反馈修正代码
            correction_prompt = f"Here is a coding problem, its initial solution, and feedback. Please provide a corrected solution.\nProblem: {prompt}\nInitial Solution: {generated_code}\nFeedback: {feedback}\nCorrected Solution:"
            corrected_code = generate_codes(model, tokenizer, correction_prompt, num_samples=1)[0]

            # 打印修正后的代码
            print(f"Corrected code for task {task_id}:")
            print(corrected_code)
            print("\n")

            # 保存修正后的代码片段
            corrected_result = {
                "task_id": task_id,
                "completion": corrected_code
            }
            save_generated_code("/data/coding/CodeUltraFeedback/codeeval/corrected_generated_codes.jsonl", corrected_result)

if __name__ == "__main__":
    main()
