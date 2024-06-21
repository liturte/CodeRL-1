from transformers import (
    LlamaTokenizer,
    LlamaForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from peft import PeftModel, PeftConfig
from core import filter_code, run_eval, fix_indents
import os
import torch

torch.cuda.set_device(2)
# TODO: move to python-dotenv
# add hugging face access token here
TOKEN = ""


@torch.inference_mode()
def generate_batch_completion(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizer, prompt, batch_size
) -> list[str]:
    input_batch = [prompt for _ in range(batch_size)]
    inputs = tokenizer(input_batch, return_tensors="pt").to(model.device)
    input_ids_cutoff = inputs.input_ids.size(dim=1)

    generated_ids = model.generate(
        **inputs,
        use_cache=True,
        max_new_tokens=2048,
        temperature=0.2,
        top_p=0.95,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,  # model has no pad token
    )

    batch_completions = tokenizer.batch_decode(
        [ids[input_ids_cutoff:] for ids in generated_ids],
        skip_special_tokens=True,
    )

    return [filter_code(fix_indents(completion)) for completion in batch_completions]


if __name__ == "__main__":
    # adjust for n = 10 etc
    num_samples_per_task = 10
    out_path = "/data/coding/CodeUltraFeedback/codeeval/feedback_apps_sft.jsonl"
    os.makedirs("/data/CodeUltraFeedback/codeeval", exist_ok=True)

    # 从本地路径加载分词器
    tokenizer = LlamaTokenizer.from_pretrained(
        "/data/coding/CodeRL/Fine_tuning_result",
    )

    # 从本地路径加载模型并加载基础权重
    base_model_path = "/data/coding/CodeUltraFeedback/models"
    adapter_model_path = "/data/coding/CodeRL/Fine_tuning_result"

    base_model = LlamaForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
    )

    # 加载适配层配置
    peft_config = PeftConfig.from_pretrained(adapter_model_path)

    # 使用peft库加载适配层权重
    model = PeftModel.from_pretrained(base_model, adapter_model_path)

    model = torch.compile(model.eval().to("cuda"))
    run_eval(
        model,
        tokenizer,
        num_samples_per_task,
        out_path,
        generate_batch_completion,
        True,
    )
