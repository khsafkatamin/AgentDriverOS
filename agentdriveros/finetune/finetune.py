import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, BitsAndBytesConfig
from transformers import DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import torch
import json


def finetune_gemma_2b_from_prompt_response(json_path, model_id="google/gemma-2b", output_dir="./gemma-2b-finetuned", num_train_epochs=3):
    # Load prompt-response dataset
    dataset = load_dataset("json", data_files=json_path, split="train")

    # Load tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    # Combine prompt + response into a single string for causal LM training
    def format_example(example):
        full_text = example["prompt"] + " " + example["response"] + tokenizer.eos_token
        return {"text": full_text}

    dataset = dataset.map(format_example)

    def tokenize(example):
        return tokenizer(example["text"], truncation=True, padding="max_length", max_length=256)

    dataset = dataset.map(tokenize, batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    # Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4"
    )

    # Load and prepare model
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")
    model = prepare_model_for_kbit_training(model)

    # Apply LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],  # Adjust if these aren't valid in Gemma
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    # Training config
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=num_train_epochs,
        learning_rate=2e-5,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        bf16=True,
        report_to="none"
    )

    # Collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator
    )

    # Train and save
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"✅ Fine-tuning complete. Model saved at: {output_dir}")


if __name__ == "__main__":
    finetune_gemma_2b_from_prompt_response(
        json_path="data/finetune/hf_finetune_planner_100.json",
        model_id="google/gemma-2b",
        output_dir="./gemma-2b-finetune",
        num_train_epochs=3
    )