# Pytorch Training using huggingface accelerate
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
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
import logging
import argparse



default_values = dict(
    accumulate_grad_batches=1,
    eval_batch_size=1,
    from_flax=False
)


@torch.no_grad()
def evaluate(accelerator, model, dataloader, global_step):
    model.eval()
    epoch_tqdm = tqdm(dataloader, disable=not accelerator.is_local_main_process, position=1, leave=False)
    losses = []
    for step, batch in enumerate(epoch_tqdm):
        ids = batch["input_ids"]
        loss = model(input_ids=ids, labels=ids).loss
        losses.append(loss)

    dist_loss = torch.stack(losses).mean()
    all_losses = accelerator.gather(dist_loss)
    eval_mean_loss = all_losses.mean().item()

    if accelerator.is_local_main_process:
        print("Eval_mean_loss", eval_mean_loss)
        accelerator.log({
            'eval/loss': eval_mean_loss
        })

    return eval_mean_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(dest="config", action="store")
    args = parser.parse_args()

    args = OmegaConf.load(args.config)
    for k, v in default_values.items():
        if k not in args:
            args[k] = v

    train_dataset = load_dataset("json", data_dir=args.data_dir, split="train", cache_dir=args.cache_dir).with_format("torch")
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    if args.get('eval_data_dir') is not None:
        eval_dataset = load_dataset("json", data_dir=args.eval_data_dir, split="train", cache_dir=args.cache_dir).with_format("torch")
        eval_dataloader = DataLoader(eval_dataset, batch_size=args.eval_batch_size, shuffle=False, drop_last=False)
    else:
        eval_dataset = None
        eval_dataloader = None


    steps_per_epoch = len(train_dataset) // (args.batch_size * args.accumulate_grad_batches)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        revision=args.get('revision'),
        from_flax=args.get('from_flax')
        )
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.learning_rate,
        betas=(0.9, 0.98)
        )
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=args.learning_rate,
        steps_per_epoch=steps_per_epoch,  
        epochs=args.num_epochs,
        anneal_strategy='linear',
        pct_start=0.01,
        final_div_factor=10
        )

    os.environ["WANDB_NAME"] = args.run_name
    accelerator = Accelerator(log_with="wandb")
    accelerator.init_trackers(
        args.project, 
        config=args,
        )

    model, optimizer, train_dataloader, lr_scheduler, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler, eval_dataloader
    )

    global_step = 0
    optimizer_step = 0


    for epoch in tqdm(range(args.num_epochs), position=0, disable=not accelerator.is_local_main_process):
        model.train()
        epoch_tqdm = tqdm(train_dataloader, disable=not accelerator.is_local_main_process, position=1, leave=False)
        
        for step, batch in enumerate(epoch_tqdm):
            ids = batch["input_ids"]
            loss = model(input_ids=ids, labels=ids).loss / args.accumulate_grad_batches
            accelerator.backward(loss)

            if (global_step + 1) % args.accumulate_grad_batches == 0:
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                optimizer_step += 1

                if accelerator.is_main_process and optimizer_step % args.logging_steps == 0:
                    metrics = {
                        'optimizer_step': optimizer_step,
                        'train/epoch': optimizer_step / steps_per_epoch,
                        'train/learning_rate': lr_scheduler.scheduler._last_lr[0],
                        'train/loss': loss.item() * args.accumulate_grad_batches
                    }
                    accelerator.log(metrics)
                    epoch_tqdm.set_description(f'loss: {loss.item() * args.accumulate_grad_batches}')
                    
            global_step += 1

        if accelerator.is_main_process:
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(f"{args.output_dir}/{args.run_name}/epoch-{epoch + 1}")

        accelerator.wait_for_everyone()

        if eval_dataloader is not None:
            evaluate(accelerator, model, eval_dataloader, optimizer_step)

    accelerator.end_training()
