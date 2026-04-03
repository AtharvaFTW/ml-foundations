import torch
from peft import LoraConfig
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer
from datasets import load_dataset

MODEL_NAME = "mistralai/Mistral-7B-v0.1"
DATASET = load_dataset("yahma/alpaca-cleaned")

bnb_config = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_quant_type = "nf4",
    bnb_4bit_compute_dtype = torch.bfloat16,
    bnb_4bit_use_double_quant = True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config = bnb_config,
    device_map = "auto"
)

peft_config = LoraConfig(
    r = 32,
    lora_alpha = 16,
    lora_dropout = 0.05,
    bias = "none",
    task_type = "CAUSAL_LM"
)

training_args = SFTConfig(
    learning_rate = 2.0e-4,
)

trainer = SFTTrainer(
    model = model,
    args = training_args,
    train_dataset = DATASET,
    peft_config = peft_config,
)

trainer.train()