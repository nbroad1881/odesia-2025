import os
import pickle
from pathlib import Path

from peft import PeftModel
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForCausalLM
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import hydra
from omegaconf import DictConfig

from dotenv import load_dotenv

loaded = load_dotenv("../.env", override=True)

if not loaded:
    raise ValueError("No .env file found")


@hydra.main(config_path="conf", config_name="config.yaml", version_base=None)
def main(cfg: DictConfig):

    output_dir = Path(cfg.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    model = PeftModel.from_pretrained(model, cfg.lora_path).merge_and_unload()

    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.base_model,
    )

    for task in cfg.tasks:
        ds = Dataset.from_parquet(str(Path(cfg.data_dir) / f"{task}_formatted.parquet"))

        def tokenize(example):
            messages = [{"role": "user", "content": example["text"]}]
            return {
                "input_ids": tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True
                ),
            }

        ds = ds.map(tokenize, batched=False, num_proc=10, remove_columns=["text"])

        responses = ds["response"]

        keep_cols = ["input_ids"]
        ds = ds.remove_columns([x for x in ds.column_names if x not in keep_cols])

        tokenizer.padding_side = "left"
        tokenizer.pad_token = cfg.pad_token
        model.config.pad_token_id = tokenizer.encode(cfg.pad_token)[0]

        collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

        all_preds = []

        mnt = 100
        for k, v in cfg.max_num_tokens.items():
            if k in task:
                mnt = v
                break

        bs = 4
        for k, v in cfg.batch_sizes.items():
            if k in task:
                bs = v
                break




        should_generate = mnt != 1

        with torch.inference_mode():
            for batch in tqdm(
                DataLoader(
                    ds, batch_size=bs, shuffle=False, collate_fn=collator, num_workers=1
                ),
                desc=f"Processing {task}",
            ):

                if should_generate:

                    preds = model.generate(
                        batch["input_ids"].to(model.device),
                        attention_mask=batch["attention_mask"].to(model.device),
                        max_new_tokens=mnt,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )

                    all_preds.extend(preds.tolist())

                else:
                    preds = model(
                        batch["input_ids"].to(model.device),
                        attention_mask=batch["attention_mask"].to(model.device),
                    ).logits

                    preds = preds[:, -1, :]

                    preds = torch.softmax(preds[:, cfg.number_ids], dim=-1)


                    all_preds.extend(preds.tolist())


        with open(output_dir / f"{task.replace('/', '_')}_preds.pkl", "wb") as f:
            pickle.dump((all_preds, responses), f)


if __name__ == "__main__":
    main()
