import torch
import datasets
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
import pprint
import axlearn.common.config as ax


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


def assert_gpu():
    assert torch.cuda.is_available(), (
        "This program requires GPUs. You can get a free GPU in Google Golab by choose the menu "
        + "Runtime -> Change runtime type and select a GPU"
    )
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")


def model(name: str):
    """We need this wrapper function for AutoModelForCausalLM.from_pretrained due to the reason
    explained in https://colab.research.google.com/drive/1WqJa2dZ7Uu98wcUlI2k1vok4r0acFE-s
    For tokenizers, axlearn.common.config works with AutoTokenizer.from_pretrained."""
    return AutoModelForCausalLM.from_pretrained(
        name, device_map="auto", torch_dtype=torch.float16, low_cpu_mem_usage=True
    )


class Finetuner:
    def __init__(self, model_cfg, tokenizer_cfg, dataset_cfg, training_args: TrainingArguments):
        self.tokenizer = tokenizer_cfg.instantiate()
        self.model = model_cfg.instantiate()
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self._preprocess(dataset_cfg.instantiate(), self.tokenizer),
            data_collator=DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False,  # We want causal language modeling, not masked
            ),
        )

    def _preprocess(self, ds, tokenizer):
        def _reformat(example):
            return {
                "text": f"Context: {example['context']}\n\n"
                + f"Question: {example['question']}\n\n"
                + f"Answer: {example['answers']['text'][0]}"
            }

        def _tokenize(examples):
            return tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=512,  # Adjust as needed
            )

        return ds.map(_reformat, remove_columns=ds.column_names).map(_tokenize, batched=True)

    def run(self, output_dir: str = "/tmp/qwen-squad-model"):
        self.trainer.train()
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)


cfg = ax.config_for_class(Finetuner).set(
    model_cfg=ax.config_for_function(model).set(name=MODEL_NAME),
    tokenizer_cfg=ax.config_for_function(AutoTokenizer.from_pretrained).set(
        pretrained_model_name_or_path=MODEL_NAME, inputs=[], kwargs={}
    ),
    dataset_cfg=ax.config_for_function(datasets.load_dataset).set(
        path="squad", split="train[:128]", config_kwargs={}
    ),
    training_args=TrainingArguments(
        output_dir="./results",
        num_train_epochs=1,  # Just 1 epoch for quick demonstration
        per_device_train_batch_size=2,  # Increased batch size for smaller model
        gradient_accumulation_steps=4,  # Reduced for smaller model
        learning_rate=5e-5,  # Slightly higher learning rate
        warmup_steps=10,
        logging_steps=10,
        save_steps=250,
        bf16=True,
        optim="adamw_torch",
        max_steps=15,  # Limit to 10 steps for quick training
        report_to="none",  # Remove if you want to use wandb or tensorboard
        gradient_checkpointing=True,  # Enable gradient checkpointing to save memory
        deepspeed=None,  # Can be enabled if deepspeed is installed
        ddp_find_unused_parameters=False,  # Use less memory
        dataloader_pin_memory=False,  # Disable pinned memory
    ),
)
pprint.pp(cfg.to_dict())

tunner = cfg.instantiate()
print(type(tunner))


tunner.run()
