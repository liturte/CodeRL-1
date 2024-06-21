import os
# 设置要使用的GPU编号
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5,6,7"  # 排除第0号GPU
import torch
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, RandomSampler
from torch.optim import AdamW



# 检查可用的GPU数量
num_gpus = torch.cuda.device_count()
print("Number of GPUs available:", num_gpus)
for i in range(num_gpus):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

# 定义数据集路径
sft_dataset_path = "/data/coding/CodeRL/outputs/test_datasets/sft_dataset.jsonl"

# 加载转换后的SFT数据集
dataset = load_dataset("json", data_files=sft_dataset_path, split='train')

# 打印数据集中的一个示例，检查数据结构
print(dataset[0])

# 划分数据集为训练集和测试集
train_test_split = dataset.train_test_split(test_size=0.1, seed=42)
datasets = DatasetDict({
    'train': train_test_split['train'],
    'test': train_test_split['test']
})

# 设置模型路径
local_model_path = "/data/coding/CodeUltraFeedback/models"

# 加载预训练模型和tokenizer
tokenizer = AutoTokenizer.from_pretrained(local_model_path)

# 设置 pad_token
tokenizer.pad_token = tokenizer.eos_token

# 加载模型
model = AutoModelForCausalLM.from_pretrained(local_model_path)

# 配置LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none"
)
model = get_peft_model(model, lora_config)

# 将模型移动到指定的GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 指定使用第0号GPU（第1号物理GPU）
model.to(device)

# 数据预处理函数
def preprocess_function(examples):
    instructions = examples['instruction']
    solutions = examples['solution']

    model_inputs = tokenizer(instructions, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(solutions, max_length=512, truncation=True, padding="max_length")["input_ids"]

    # 设置标签，将填充的部分设为-100以便在计算损失时忽略它们
    for i in range(len(labels)):
        labels[i] = [(l if l != tokenizer.pad_token_id else -100) for l in labels[i]]

    model_inputs["labels"] = labels

    return model_inputs

# 应用数据预处理
tokenized_datasets = datasets.map(preprocess_function, batched=True, remove_columns=["instruction", "solution"])

# 创建数据加载器
train_sampler = RandomSampler(tokenized_datasets['train'])
train_dataloader = DataLoader(tokenized_datasets['train'], batch_size=1, sampler=train_sampler)

eval_sampler = RandomSampler(tokenized_datasets['test'])
eval_dataloader = DataLoader(tokenized_datasets['test'], batch_size=1, sampler=eval_sampler)

# 定义优化器
optimizer = AdamW(model.parameters(), lr=2e-4)

# 定义训练参数
training_args = TrainingArguments(
    output_dir="/data/coding/CodeRL/Fine_tuning_result",
    evaluation_strategy="epoch",
    learning_rate=2e-4,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='/data/coding/CodeRL/logs',
    logging_steps=10,
    fp16=True,
)

# 微调模型
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
    optimizers=(optimizer, None),
)

trainer.train()

# 保存微调后的模型权重和tokenizer
model.save_pretrained("/data/coding/CodeRL/Fine_tuning_result")
tokenizer.save_pretrained("/data/coding/CodeRL/Fine_tuning_result")

print("微调完成，模型已保存。")

# 加载微调后的模型和tokenizer
fine_tuned_model_path = "/data/coding/CodeRL/Fine_tuning_result"
model = AutoModelForCausalLM.from_pretrained(fine_tuned_model_path)
tokenizer = AutoTokenizer.from_pretrained(fine_tuned_model_path)

# 将模型移动到指定的GPU
model.to(device)

# 简单测试
def generate_code(instruction):
    inputs = tokenizer(f"Instruction: {instruction}", return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    outputs = model.generate(**inputs, max_length=512, num_return_sequences=1)
    generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_code

# 测试一些样例
test_instructions = [
    "Write a function to add two numbers.",
    "Create a class for a simple calculator.",
]

for instruction in test_instructions:
    print(f"Instruction: {instruction}")
    print("Generated Code:", generate_code(instruction))
