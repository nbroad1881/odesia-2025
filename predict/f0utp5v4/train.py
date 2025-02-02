from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf
import hydra
import wandb
import shutil
import os
from functools import partial
from pathlib import Path
from trl import (
    SFTTrainer,
    ModelConfig,
    get_quantization_config,
    get_kbit_device_map,
    get_peft_config,
    DataCollatorForCompletionOnlyLM,
)
from dotenv import load_dotenv
from peft import (
    get_peft_model,
    prepare_model_for_kbit_training,
    AutoPeftModelForSequenceClassification,
)

# from utils import add_metric_to_card

loaded = load_dotenv("../.env", override=True)

if not loaded:
    raise ValueError("Failed to load .env file")


def tokenize(example, tokenizer):
    ids = tokenizer.apply_chat_template([
        {"role": "user", "content": example["text"]},
        {"role": "assistant", "content": example["response"]},
    ])

    return {
        "input_ids": ids,
    }


@hydra.main(config_path="conf", config_name="q7b-4bit")
def main(cfg: DictConfig):

    cfg.time_start = "_".join(str(Path.cwd()).rsplit("/", 2)[-2:])

    if cfg.DEBUG:
        cfg.model_config.model_name_or_path = cfg.debug_model

    script_args = cfg.script_args
    training_args = TrainingArguments(**OmegaConf.to_container(cfg.training_args))
    model_config = ModelConfig(**OmegaConf.to_container(cfg.model_config))

    if training_args.process_index == 0:

        if cfg.eval_only or training_args.resume_from_checkpoint is not None:
            wandb_id = cfg.wandb_id
            resume = "must"
            config = None
        else:
            wandb_id = None
            resume = None
            config = OmegaConf.to_container(cfg)

        wandb.init(config=config, id=wandb_id, resume=resume)
        # copy current file to output, so it gets saved to hub
        shutil.copy(
            Path(__file__).resolve(),
            Path(training_args.output_dir) / Path(__file__).name,
        )

        shutil.copy(
            Path(__file__).resolve().parent / "utils.py",
            Path(training_args.output_dir) / "utils.py",
        )

    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=model_config.torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
        cache_dir=os.environ["HF_HUB_CACHE"],
    )

    peft_config = get_peft_config(model_config)

    if training_args.use_liger_kernel:
        from liger_kernel.transformers import (
            apply_liger_kernel_to_qwen2,
            apply_liger_kernel_to_llama,
            apply_liger_kernel_to_mistral,
        )

        apply_liger_kernel_to_qwen2()
        apply_liger_kernel_to_llama()
        apply_liger_kernel_to_mistral()
    if cfg.eval_only:

        model = AutoPeftModelForSequenceClassification.from_pretrained(
            model_config.model_name_or_path,
            **model_kwargs,
            token=os.environ["HF_WRITE_PERSONAL"],
        )

        if cfg.merge_adapters:
            model = model.merge_and_unload()

    else:

        model = AutoModelForCausalLM.from_pretrained(
            model_config.model_name_or_path,
            **model_kwargs,
            token=os.environ["HF_GATED"],
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path,
        use_fast=True,
        token=os.environ["HF_GATED"],
    )

    tokenizer.padding_side = "left"
    tokenizer.pad_token = cfg.pad_token


    if not cfg.eval_only and model_config.load_in_4bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=training_args.gradient_checkpointing,
            gradient_checkpointing_kwargs=training_args.gradient_checkpointing_kwargs,
        )

    elif not cfg.eval_only and training_args.gradient_checkpointing:
        model.enable_input_require_grads()

    if not cfg.eval_only:
        model = get_peft_model(model, peft_config)

    with training_args.main_process_first():
        ds = load_dataset(
            script_args.dataset_name,
            script_args.config,
            token=os.environ["HF_WRITE_PERSONAL"],
        )

        if cfg.DEBUG:
            ds[cfg.train_split_name] = (
                ds[cfg.train_split_name].shuffle().select(range(100))
            )
            ds[cfg.val_split_name] = ds[cfg.val_split_name].shuffle().select(range(100))

        if not cfg.eval_only:
            ds[cfg.val_split_name] = ds[cfg.val_split_name].shuffle().select(range(500))

        ds = ds.map(tokenize, fn_kwargs={"tokenizer": tokenizer}, num_proc=cfg.num_proc, remove_columns=ds["train"].column_names)

    collator = DataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=16,
        response_template=cfg.response_template_ids
    )

    if training_args.process_index == 0:
        group = os.environ["WANDB_RUN_GROUP"]
        training_args.hub_model_id = f"nbroad/nbroad-odesia-{group}-{wandb.run.id}"
        training_args.hub_token = os.environ["HF_WRITE_PERSONAL"]

    prefix = ""

    if cfg.eval_only:
        if "awq" in model_config.model_name_or_path.lower():
            prefix = "awq_"
        if model_config.load_in_4bit:
            prefix += "int4_"
        elif model_config.torch_dtype == "bfloat16":
            prefix += "bf16_"
        elif model_config.torch_dtype == "float16":
            prefix += "fp16_"

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=(
            ds[cfg.val_split_name] if training_args.eval_strategy != "no" else None
        ),
        processing_class=tokenizer,
        data_collator=collator,
        # compute_metrics=partial(compute_metrics, prefix=prefix),
    )

    if training_args.process_index == 0:

        trainer.model.config.update(
            {
                "wandb_id": wandb.run.id,
                "fold": cfg.fold,
                "group": group,
                "dataset": script_args.dataset_name,
            }
        )

    if not cfg.eval_only:
        if training_args.resume_from_checkpoint is not None:
            os.chdir(Path(training_args.resume_from_checkpoint).parent)
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    else:
        metrics = trainer.evaluate()

        # if training_args.process_index == 0:
            # met = [x for x in metrics if "accuracy" in x][0]

            # result = add_metric_to_card(
            #     repo=training_args.hub_model_id,
            #     metrics_pretty_name=met,
            #     metrics_value=metrics[met],
            #     dataset_id=script_args.dataset_name,
            #     dataset_split=cfg.val_split_name,
            #     model_path=model_config.model_name_or_path,
            #     model_dtype=model_config.torch_dtype,
            #     token=os.environ["HF_WRITE_PERSONAL"],
            # )
            # print(result)

    if not cfg.eval_only:
        # Save and push to hub
        trainer.save_model(training_args.output_dir)
        if training_args.push_to_hub:
            trainer.push_to_hub(
                dataset_name=script_args.dataset_name,
                model_name=model_config.model_name_or_path,
                tags=cfg.hub_repo_tags,
            )

    if training_args.process_index == 0:
        wandb.finish()


if __name__ == "__main__":
    main()
