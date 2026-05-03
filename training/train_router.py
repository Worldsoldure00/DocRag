"""
Fine-tune Phi-4-mini as the domain router using Unsloth + QLoRA.
Designed to run on Google Colab Pro (A100 40GB) or local RTX 4070 (tight but feasible).

Usage (Colab):
    !pip install unsloth
    !python training/train_router.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import json
import logging
from pathlib import Path

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

TRAIN_DATA = config.BASE_DIR / "training" / "data" / "router_train.json"
OUTPUT_DIR = config.MODELS_DIR / "router_adapter"


def train():
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        raise ImportError("Install unsloth: pip install unsloth")

    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset

    log.info("Loading base model: %s", config.ROUTER_BASE_MODEL)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.ROUTER_BASE_MODEL,
        max_seq_length=512,      # router only needs short sequences
        load_in_4bit=True,
        token=config.HF_TOKEN,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=config.LORA_RANK,
        lora_alpha=config.LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=config.LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    log.info("Loading training data: %s", TRAIN_DATA)
    with open(TRAIN_DATA, encoding="utf-8") as f:
        data = json.load(f)
    dataset = Dataset.from_list(data)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            output_dir=str(OUTPUT_DIR),
            per_device_train_batch_size=8,   # router uses short sequences, can use larger batch
            gradient_accumulation_steps=2,
            num_train_epochs=5,              # more epochs for small classification task
            learning_rate=config.LEARNING_RATE,
            fp16=True,
            logging_steps=10,
            save_strategy="epoch",
            warmup_steps=20,
            dataset_text_field="text",
            max_seq_length=512,
        ),
    )

    log.info("Starting router fine-tuning...")
    trainer.train()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    log.info("Router adapter saved → %s", OUTPUT_DIR)

    # Also save merged model for GGUF conversion
    merged_dir = config.MODELS_DIR / "router_merged"
    model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")
    log.info("Merged router model saved → %s (convert to GGUF for Ollama)", merged_dir)


if __name__ == "__main__":
    train()
