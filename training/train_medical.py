"""
Fine-tune BioMistral-7B as the medical expert using Unsloth + QLoRA.
BioMistral is already pretrained on PubMed — fine-tuning gives fast convergence.
Run on Google Colab Pro A100 (40GB) — estimated ~2-3 hours for 1500 examples.

Usage (Colab):
    !pip install unsloth
    !python training/train_medical.py
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

TRAIN_DATA = config.BASE_DIR / "training" / "data" / "medical_sft.json"
OUTPUT_DIR = config.MODELS_DIR / "medical_adapter"


def train():
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        raise ImportError("Install unsloth: pip install unsloth")

    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset

    log.info("Loading base model: %s", config.MEDICAL_BASE_MODEL)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.MEDICAL_BASE_MODEL,
        max_seq_length=config.MAX_SEQ_LENGTH,
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
            output_dir=str(OUTPUT_DIR / "checkpoints"),
            per_device_train_batch_size=config.TRAIN_BATCH_SIZE,
            gradient_accumulation_steps=config.GRAD_ACCUMULATION,
            num_train_epochs=config.NUM_EPOCHS,
            learning_rate=config.LEARNING_RATE,
            fp16=True,
            logging_steps=25,
            save_strategy="epoch",
            warmup_steps=config.WARMUP_STEPS,
            lr_scheduler_type="cosine",
            dataset_text_field="text",
            max_seq_length=config.MAX_SEQ_LENGTH,
        ),
    )

    log.info("Starting medical expert fine-tuning (~2-3h on A100)...")
    trainer.train()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    log.info("Medical adapter saved → %s", OUTPUT_DIR)

    merged_dir = config.MODELS_DIR / "medical_merged"
    model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")
    log.info("Merged medical model → %s", merged_dir)

    log.info("To serve with Ollama:")
    log.info("  1. Convert: python -m llama_cpp.convert_hf_to_gguf %s --outtype q4_k_m", merged_dir)
    log.info("  2. Create Modelfile with FROM pointing to the .gguf file")
    log.info("  3. ollama create %s -f Modelfile", config.OLLAMA_MEDICAL)


if __name__ == "__main__":
    train()
