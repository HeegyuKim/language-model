# Pytorch Training using huggingface accelerate
from transformers import (
    HfArgumentParser,
    DataCollatorForSeq2Seq,
    DataCollatorWithPadding,
    set_seed
)
from utils.collator import SequenceClassificationCollator
from utils.arguments import TrainingArguments, DataTrainingArguments, ModelArguments
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer

from typing import List, Dict, Any
from omegaconf import OmegaConf
import fire
from datasets import disable_caching, load_dataset
from accelerate import Accelerator
from tqdm.auto import tqdm
from accelerate.logging import get_logger
import os
import evaluate
from pprint import pprint


eval_accuracy = evaluate.load("accuracy")

def train(
    args, accelerator, model, optimizer, lr_scheduler, train_dataloader, eval_dataloader
):

    global_step = 0
    optimizer_step = 0

    for epoch in tqdm(
        range(args.num_train_epochs),
        position=0,
        disable=not accelerator.is_local_main_process,
    ):
        model.train()
        epoch_tqdm = tqdm(
            train_dataloader,
            disable=not accelerator.is_local_main_process,
            position=1,
            leave=False,
        )

        for step, batch in enumerate(epoch_tqdm):
            loss = model(**batch).loss / args.gradient_accumulation_steps
            accelerator.backward(loss)

            if (global_step + 1) % args.gradient_accumulation_steps == 0:
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()


                if (
                    accelerator.is_main_process
                    and optimizer_step % args.logging_steps == 0
                ):
                    metrics = {
                        "optimizer_step": optimizer_step,
                        "train/learning_rate": lr_scheduler.scheduler._last_lr[0],
                        "train/loss": loss.item() * args.gradient_accumulation_steps,
                    }
                    accelerator.log(metrics)

                optimizer_step += 1
                epoch_tqdm.set_description(
                    f"loss: {loss.item() * args.gradient_accumulation_steps}"
                )

                if (
                    args.eval_strategy == 'steps'
                    and optimizer_step % args.eval_steps == 0
                ):
                    evaluate(accelerator, model, eval_dataloader)

            global_step += 1

        if args.save_strategy == "epoch":
            if accelerator.is_main_process:
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(
                    f"{args.output_dir}/{args.run_name}/epoch-{epoch + 1}"
                )

            accelerator.wait_for_everyone()

        if eval_dataloader is not None and args.eval_strategy == 'epoch':
            evaluate(accelerator, model, eval_dataloader)


def collate_dictlist(dl):
    from collections import defaultdict
    out = defaultdict(list)

    for d in dl:
        for k, v in d.items():
            out[k].append(v)
    
    return out

@torch.no_grad()
def evaluate(accelerator, model, dataloader):
    model.eval()
    epoch_tqdm = tqdm(
        dataloader,
        disable=not accelerator.is_local_main_process,
        position=1,
        leave=False,
    )
    step_outputs = []
    for step, batch in enumerate(epoch_tqdm):
        out = model(**batch)
        loss, logits = out.loss, out.logits

        step_outputs.append({
            'loss': loss,
            'logits': logits,
            'labels': batch['labels']
        })

    eval_outputs = accelerator.gather_for_metrics(step_outputs)
    # eval_mean_loss = all_losses.mean().item()

    if accelerator.is_local_main_process:
        eval_outputs = collate_dictlist(eval_outputs)
        preds = torch.stack(eval_outputs['logits']).view(-1, 2).argmax(-1)
        labels = torch.stack(eval_outputs['labels']).view(-1)
        eval_mean_loss = torch.stack(eval_outputs['loss']).mean().item()
        eval_results = {
            "eval/loss": eval_mean_loss,
            "eval/accuracy": eval_accuracy.compute(predictions=preds, references=labels)['accuracy']
        }
        pprint("evaluation result")
        pprint(eval_results)
        accelerator.log(eval_results)

    model.train()


def main():
    parser = HfArgumentParser(
        (TrainingArguments, DataTrainingArguments, ModelArguments)
    )
    training_args, data_args, model_args = parser.parse_args_into_dataclasses()
    args = parser.parse_args()

    set_seed(args.seed)

    os.environ["WANDB_NAME"] = args.run_name
    accelerator = Accelerator(log_with="wandb")
    accelerator.init_trackers(
        args.project,
        config=args,
    )

    dataset = load_dataset(args.dataset_name, args.dataset_config_name)

    model = model_args.get_model()
    tokenizer = model_args.get_tokenizer()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def encode_data(x):
        ids = tokenizer.encode(x["document"], truncation=True, max_length=args.max_sequence_length)
        out = {"input_ids": ids, "attention_mask": [1] * len(ids)}
        out["label"] = x["label"]
        return out

    with accelerator.local_main_process_first():
        mapped_dataset = dataset.map(
            encode_data, remove_columns=dataset["train"].column_names
        )

    collator = SequenceClassificationCollator(
        tokenizer=tokenizer,
        max_length=args.max_sequence_length,
        pad_to_multiple_of=8,
        padding="max_length",
        return_tensors="pt",
    )
    train_dataloader = DataLoader(
        mapped_dataset["train"],
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collator,
    )

    if args.do_eval:
        eval_dataloader = DataLoader(
            mapped_dataset["test"],
            batch_size=args.per_device_eval_batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=collator,
        )
    else:
        eval_dataloader = None

    steps_per_epoch = len(dataset["train"]) // (
        args.per_device_train_batch_size
        * args.gradient_accumulation_steps
    )

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.learning_rate,
        steps_per_epoch=steps_per_epoch,
        epochs=args.num_train_epochs,
        anneal_strategy="linear",
        pct_start=0.01,
        final_div_factor=10,
    )
    (
        model,
        optimizer,
        train_dataloader,
        lr_scheduler,
        eval_dataloader,
    ) = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler, eval_dataloader
    )

    accelerator.register_for_checkpointing(lr_scheduler)

    if args.do_train:
        train(
            args,
            accelerator,
            model,
            optimizer,
            lr_scheduler,
            train_dataloader,
            eval_dataloader,
        )
    elif args.do_eval:
        evaluate(accelerator, model, eval_dataloader)
    accelerator.end_training()
