import json
import re
import tqdm
import io
import contextlib
from tenacity import retry, stop_after_attempt, wait_random_exponential
from concurrent.futures import ThreadPoolExecutor

# 设置 API 密钥和 API 基础 URL
DEEPSEEK_API_KEY = "sk-b7a1e5676ec34a46b71e589d283c8d84"
DEEPSEEK_API_BASE = "https://api.deepseek.com/v1"
GPT_API_KEY = "sk-8XGqiRc9787Q19pnlyrGu5L6Gm7z81fIv2Rv1JWlmmCs6GkY"
GPT_API_BASE = "https://api.chatanywhere.tech/v1"

# 指定 HumanEval 数据集路径和输出文件路径
HUMAN_EVAL = 'D:/Pycharm/CodeRL/my_code_eval/human-eval/data/HumanEval.jsonl'
OUT_FILE = 'D:/Pycharm/CodeRL/my_code_eval/human-eval/data/new_results-{}.jsonl'

pattern = re.compile(r'```(?:[Pp]ython|[Pp]y)\s*([\s\S]+?)\s*```')

COMMON_IMPORTS = """
import math
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any, Optional
"""

# 初始化 DeepSeek 和 GPT 客户端
from openai import OpenAI

deepseek_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_API_BASE)
gpt_client = OpenAI(api_key=GPT_API_KEY, base_url=GPT_API_BASE)

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def get_initial_completion(prompt, model='deepseek-coder'):
    completion = deepseek_client.chat.completions.create(
        model="deepseek-coder",
        messages=[
            {"role": "system", "content": "You are an intelligent programmer. You must complete the python function given to you by the user. And you must follow the format they present when giving your answer!"},
            {"role": "user", "content": prompt}
        ],
        stream=False,
        temperature=0
    )
    result = completion.choices[0].message.content
    match = pattern.search(result)

    if match:
        python_code = match.group(1)
    else:
        python_code = result

    return python_code

def generate_test_cases(prompt, code):
    test_cases_prompt = f"Here is a problem statement:\n{prompt}\n\nHere is a solution:\n{code}\n\nBased on the problem statement and solution, generate some test cases to verify the correctness of the solution. Provide the test cases in the following format:\n\nTest Case:\nExpected Output:"
    completion = gpt_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a code reviewer. You need to generate test cases based on the problem statement and provided solution."},
            {"role": "user", "content": test_cases_prompt}
        ],
        stream=False,        temperature=0.3
    )
    result = completion.choices[0].message.content
    return result

def parse_test_cases(test_cases):
    test_case_list = []
    for test_case in test_cases.split('\n\n'):
        if 'Test Case:' in test_case and 'Expected Output:' in test_case:
            parts = test_case.split('\n')
            test_case_dict = {
                "Test Case": parts[0].replace('Test Case:', '').strip(),
                "Expected Output": parts[1].replace('Expected Output:', '').strip()
            }
            test_case_list.append(test_case_dict)
    return test_case_list

def run_tests(code, test_cases):
    results = []
    exec_globals = {}
    exec(COMMON_IMPORTS + code, exec_globals)

    for test_case in test_cases:
        try:
            test_code = test_case["Test Case"]
            expected_output = test_case["Expected Output"]

            with contextlib.redirect_stdout(io.StringIO()) as f:
                exec(test_code, exec_globals)
                output = f.getvalue().strip()

            result = {
                "test_case": test_code,
                "expected_output": expected_output,
                "actual_output": output,
                "passed": expected_output == output
            }
        except Exception as e:
            result = {
                "test_case": test_code,
                "expected_output": expected_output,
                "actual_output": str(e),
                "passed": False
            }

        results.append(result)

    return results

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def get_evaluation(prompt, code, test_cases, test_results, model='gpt-3.5-turbo'):
    eval_prompt = f"Here is the initial problem statement:\n{prompt}\n\nHere is the initial code solution:\n{code}\n\nHere are the test cases:\n{test_cases}\n\nHere are the test results:\n{test_results}\n\nBased on this information, provide a detailed evaluation and feedback."
    completion = gpt_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a code reviewer. You need to evaluate the provided code and give detailed feedback."},
            {"role": "user", "content": eval_prompt}
        ],
        stream=False,
        temperature=0.3
    )
    result = completion.choices[0].message.content
    return result

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def get_final_completion(prompt, code, evaluation, model='deepseek-coder'):
    completion = deepseek_client.chat.completions.create(
        model="deepseek-coder",
        messages=[
            {"role": "system", "content": "You are an intelligent programmer. You must refine the provided python function based on the feedback and evaluation."},
            {"role": "user", "content": f"Here is the initial problem statement:\n{prompt}\n\nHere is the initial code solution:\n{code}\n\nHere is the evaluation and feedback:\n{evaluation}\n\nBased on this information, only provide the final refined code solution, do not output test cases"}
        ],
        stream=False,
        temperature=0
    )
    result = completion.choices[0].message.content
    match = pattern.search(result)

    if match:
        python_code = match.group(1)
    else:
        python_code = result
    print("final code:")
    print(python_code)
    return python_code

def iter_hval():
    with open(HUMAN_EVAL) as f:
        for line in f:
            yield json.loads(line)

def process_command(command, temperature):
    task_id, prompt = command
    initial_code = get_initial_completion(prompt, model='deepseek-coder')
    test_cases = generate_test_cases(prompt, initial_code)
    test_case_list = parse_test_cases(test_cases)
    test_results = run_tests(initial_code, test_case_list)
    evaluation = get_evaluation(prompt, initial_code, test_cases, test_results, model='gpt-3.5-turbo')
    final_code = get_final_completion(prompt, initial_code, evaluation, model='deepseek-coder')
    return {'task_id': task_id, 'completion': final_code}

def get_results():
    out_file = OUT_FILE.format("deepseek-coder")

    with open(out_file, 'w') as f:
        pass

    batch_size = 15
    batch = []
    total_tasks = sum(1 for _ in iter_hval())
    with tqdm.tqdm(total=total_tasks) as progress_bar:
        for line in iter_hval():
            prompt = line['prompt']
            task_id = line['task_id']
            batch.append((task_id, prompt))

            if len(batch) == batch_size:
                with ThreadPoolExecutor() as executor:
                    futures = [executor.submit(process_command, command, 0) for command in batch]
                    results = [future.result() for future in futures]

                with open(out_file, 'a') as out_f:
                    for out in results:
                        out_f.write(json.dumps(out) + '\n')

                batch = []
                progress_bar.update(batch_size)

        if batch:
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(process_command, command, 0) for command in batch]
                results = [future.result() for future in futures]

            with open(out_file, 'a') as out_f:
                for out in results:
                    out_f.write(json.dumps(out) + '\n')

            progress_bar.update(len(batch))

if __name__ == '__main__':
    get_results()

    out_f = OUT_FILE.format("deepseek-coder")
    print(f'Tests complete at: {out_f}')
