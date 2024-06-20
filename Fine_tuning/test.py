import logging
import os
import random

import torch
import transformers
from datasets import load_dataset, DatasetDict
from rich.logging import RichHandler
from transformers import set_seed, TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel

from alignment import (
    DataArguments,
    SFTConfig,
    H4ArgumentParser,
    ModelArguments,
    maybe_insert_system_message
)

logger = logging.getLogger("rich")

def main():
    # 直接在代码中指定参数
    model_name_or_path = "/data/coding/CodeUltraFeedback/models"
    dataset_path = "/data/coding/CodeRL/outputs/test_datasets/sft_dataset.jsonl"
    output_dir = "/data/coding/CodeRL/Fine_tuning_result"
    seed = 42
    learning_rate = 2e-4  # 设置学习率
    num_train_epochs = 2  # 设置训练轮数

    logging.basicConfig(
        level="NOTSET",
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler()]
    )
    log_level = logging.INFO
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.info(f"Model parameters: {model_name_or_path}")
    logger.info(f"Data parameters: {dataset_path}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Seed: {seed}")

    set_seed(seed)

    # 确定使用的设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    ###############
    # 加载数据集
    ###############
    dataset = load_dataset("json", data_files=dataset_path, split='train')

    # 打印数据集中的一个示例，检查数据结构
    print(dataset[0])

    # 划分数据集为训练集和测试集
    train_test_split = dataset.train_test_split(test_size=0.1, seed=42)
    datasets = DatasetDict({
        'train': train_test_split['train'],
        'test': train_test_split['test']
    })

    logger.info(f"训练集样本数: {len(datasets['train'])}")
    logger.info(f"测试集样本数: {len(datasets['test'])}")

    #######################
    # 加载预训练模型
    #######################
    logger.info("*** 加载预训练模型 ***")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name_or_path,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=False
    )

    if "runs/" not in model_name_or_path:
        logger.info("初始化新的 LoRA 适配器模块。")
        model = FastLanguageModel.get_peft_model(
            model,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            bias="none",
            use_gradient_checkpointing=False,
        )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    tokenizer.truncation_side = "left"

    model = model.to(device)  # 确保模型在同一设备上

    #####################
    # 数据预处理函数
    #####################
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
                inputs.append(f"{context} Question: {question}")
                targets.append(answer)

        model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
        labels = tokenizer(targets, max_length=512, truncation=True, padding="max_length")["input_ids"]

        if len(model_inputs["input_ids"]) != len(labels):
            print(f"Warning: Length mismatch between inputs and labels for context: {contexts}")
            print(f"Inputs length: {len(model_inputs['input_ids'])}, Labels length: {len(labels)}")

        model_inputs["labels"] = labels
        return model_inputs

    # 应用数据预处理
    tokenized_datasets = datasets.map(preprocess_function, batched=True, remove_columns=datasets['train'].column_names)

    ########################
    # 初始化Trainer
    ########################
    # 计算warmup steps
    total_steps = (len(tokenized_datasets['train']) // 1) * num_train_epochs  # 假设每个设备的batch size为1
    warmup_steps = int(0.1 * total_steps)  # 设置为10%

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        num_train_epochs=num_train_epochs,
        logging_dir='./logs',
        logging_steps=10,
        save_steps=100,
        seed=seed,
        remove_unused_columns=False,  # 避免AttributeError
        fp16=True,  # 启用混合精度训练
        learning_rate=learning_rate,
        warmup_steps=warmup_steps
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['test'],
        dataset_text_field="text",
        max_seq_length=2048,
        tokenizer=tokenizer,
        packing=True,
    )

    ###############
    # 训练循环
    ###############
    logger.info("*** 开始训练 ***")
    train_result = trainer.train()
    metrics = train_result.metrics
    metrics["train_samples"] = len(tokenized_datasets['train'])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("*** 训练完成 ***")

    ##########
    # 评估
    ##########
    if training_args.do_eval:
        logger.info("*** 开始评估 ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(tokenized_datasets['test'])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    ##################################
    # 保存模型并创建模型卡
    ##################################
    logger.info("*** 保存模型 ***")
    trainer.save_model(output_dir)
    logger.info(f"模型保存至 {output_dir}")

    logger.info("*** 训练完成 ***")

if __name__ == "__main__":
    main()
