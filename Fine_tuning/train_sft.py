from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
import json

# 定义数据集路径
sft_dataset_path = "/data/coding/CodeRL/outputs/test_datasets/sft_dataset.jsonl"

# 加载转换后的数据集
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
    target_modules=["q_proj", "k_proj"],
    lora_dropout=0.05,
    bias="none"
)
model = get_peft_model(model, lora_config)

# 分块处理函数
def chunk_sequence(sequence, chunk_size):
    """将序列分成多个块，每个块的长度不超过 chunk_size。"""
    return [sequence[i:i + chunk_size] for i in range(0, len(sequence), chunk_size)]

# 数据预处理函数
def preprocess_function(examples):
    inputs = []
    targets = []

    contexts = examples['context']
    qa_pairs_list = examples['qa_pairs']

    for i in range(len(contexts)):
        context = contexts[i]
        qa_pairs = qa_pairs_list[i]
        for qa in qa_pairs:
            question = qa['question']
            answer = qa['answer']
            full_input = f"{context} Question: {question}"
            full_target = answer

            # 将长序列分成多个块
            input_chunks = chunk_sequence(full_input, 512)
            target_chunks = chunk_sequence(full_target, 512)

            # 过滤掉空的块
            input_chunks = [chunk for chunk in input_chunks if chunk]
            target_chunks = [chunk for chunk in target_chunks if chunk]

            # 调试信息：打印分块后的长度
            print(f"Input chunks: {len(input_chunks)}, Target chunks: {len(target_chunks)}")

            inputs.extend(input_chunks)
            targets.extend(target_chunks)

    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=512, truncation=True, padding="max_length")["input_ids"]

    # 确保 model_inputs 和 labels 长度一致
    if len(model_inputs["input_ids"]) == len(labels):
        model_inputs["labels"] = labels
    else:
        # 处理长度不一致的情况
        min_length = min(len(model_inputs["input_ids"]), len(labels))
        model_inputs["input_ids"] = model_inputs["input_ids"][:min_length]
        model_inputs["attention_mask"] = model_inputs["attention_mask"][:min_length]
        model_inputs["labels"] = labels[:min_length]

    # 调试信息：打印最终的输入和标签长度
    print(f"Processed inputs: {len(model_inputs['input_ids'])}, Processed labels: {len(model_inputs['labels'])}")

    return model_inputs

# 应用数据预处理
try:
    tokenized_datasets = datasets.map(preprocess_function, batched=True, remove_columns=["context", "qa_pairs"])
except Exception as e:
    print(f"Error during preprocessing: {e}")

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
try:
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['test'],
    )

    trainer.train()
except Exception as e:
    print(f"Error during training: {e}")

# 保存微调后的模型
try:
    model.save_pretrained("/data/coding/CodeRL/Fine_tuning_result")
    tokenizer.save_pretrained("/data/coding/CodeRL/Fine_tuning_result")

    print("微调完成，模型已保存。")
except Exception as e:
    print(f"Error during model saving: {e}")
