import json
import os
import re
import tqdm
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

# 设置 API 密钥
API_KEY = "sk-b7a1e5676ec34a46b71e589d283c8d84"  # 将此处替换为你的 DeepSeek API 密钥
API_BASE = "https://api.deepseek.com"  # 将此处替换为你的 DeepSeek API 基础 URL

# 初始化 OpenAI 客户端
client = OpenAI(api_key=API_KEY, base_url=API_BASE)

# 指定 HumanEval 数据集路径和输出文件路径
HUMAN_EVAL = '/root/autodl-tmp/code-eval/human-eval/data/HumanEval.jsonl'
OUT_FILE = '/root/autodl-tmp/result/results-{}.jsonl'

pattern = re.compile(r'```(?:[Pp]ython|[Pp]y)\s*([\s\S]+?)\s*```')

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def get_completion(prompt, model='deepseek-coder'):
    completion = client.chat.completions.create(
        model="deepseek-coder",
        messages=[
            {"role": "system", "content": "You are an intelligent programmer. You must complete the python function given to you by the user. And you must follow the format they present when giving your answer!"},
            {"role": "user", "content": prompt}
        ],
        stream=False,  # 确保 stream 设置为 False
        temperature=0
    )
    result = completion.choices[0].message.content
    match = pattern.search(result)
    
    if match:
        python_code = match.group(1)
    else:
        python_code = result

    return python_code

def iter_hval():
    with open(HUMAN_EVAL) as f:
        for line in f:
            yield json.loads(line)

def process_command(command, model):
    task_id, prompt = command
    completion = get_completion(prompt, model=model)
    return {'task_id': task_id, 'completion': completion}

def get_results(model='deepseek-code'):
    out_file = OUT_FILE.format(model)

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
                    futures = [executor.submit(process_command, command, model) for command in batch]
                    results = [future.result() for future in futures]

                with open(out_file, 'a') as out_f:
                    for out in results:
                        out_f.write(json.dumps(out) + '\n')

                batch = []
                progress_bar.update(batch_size)

        if batch:
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(process_command, command, model) for command in batch]
                results = [future.result() for future in futures]

            with open(out_file, 'a') as out_f:
                for out in results:
                    out_f.write(json.dumps(out) + '\n')

            progress_bar.update(len(batch))

if __name__ == '__main__':
    model = 'deepseek-code'  # 确认这个是正确的模型名称
    get_results(model=model)

    out_f = OUT_FILE.format(model)
    print(f'Tests complete at: {out_f}')
