import torch
from diffusers import StableDiffusionPipeline
from diffusers import DPMSolverMultistepScheduler
from accelerate import Accelerator
from pathlib import Path

# 参数
model_id = "runwayml/stable-diffusion-v1-5"
output_dir = "./lora-pet"
dataset_path = "./dataset/pet"

# 初始化 Accelerator
accelerator = Accelerator()
device = accelerator.device

# 加载基础模型
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16
).to(device)

pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

# LoRA 配置
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=16,  # rank
    lora_alpha=32,
    target_modules=["to_q", "to_k", "to_v"],  # attention 模块
    lora_dropout=0.1,
    bias="none",
)
pipe.unet = get_peft_model(pipe.unet, config)

# 数据集 (简单示例: 图像文件夹)
from datasets import load_dataset

dataset = load_dataset("imagefolder", data_dir=dataset_path)

# 训练器
from diffusers import DDPMScheduler
from transformers import TrainingArguments, Trainer

args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=5,
    learning_rate=1e-4,
    fp16=True,
    save_steps=100,
    logging_dir="./logs",
    remove_unused_columns=False,
)


def collate_fn(examples):
    pixel_values = [example["image"].convert("RGB") for example in examples]
    return {"pixel_values": pixel_values}


trainer = Trainer(
    model=pipe.unet,
    args=args,
    train_dataset=dataset["train"],
    data_collator=collate_fn,
)

trainer.train()
pipe.save_pretrained(output_dir)

print(f"LoRA 模型已保存到 {output_dir}")
