# Pytorch Training using huggingface accelerate
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
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


default_values = dict(
    accumulate_grad_batches=1
)

def main():
    args = OmegaConf.load(f"config/ajoublue-gpt2-base.yaml")
    for k, v in default_values.items():
        if k not in args:
            args[k] = v

    # dataset = load_dataset("json", data_dir=args.data_dir, split="train", cache_dir="/data/.cache").with_format("torch")
    dataset = load_dataset("json", data_files=["/data2/v1-vocab51k-block1024/heegyu__kowikitext.jsonl"], split="train", cache_dir="/data2/.cache").with_format("torch")
    print("data total", len(dataset), "blocks")
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    steps_per_epoch = len(dataset) // (args.batch_size * 8 * args.accumulate_grad_batches)

    config = AutoConfig.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_config(config)
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

    accelerator = Accelerator()
    # accelerator.init_trackers(args.project, config=args)

    model, optimizer, train_dataloader  = accelerator.prepare(
        model, optimizer, dataloader
    )

    global_step = 0
    optimizer_step = 0

    logger = get_logger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    if accelerator.is_main_process:
        logger.info(args)


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
                        'train/step': optimizer_step,
                        'train/epoch': epoch + optimizer_step / steps_per_epoch,
                        'train/learning_rate': lr_scheduler._last_lr,
                        'train/loss': loss.item() * args.accumulate_grad_batches
                    }
                    logger.info(str(metrics))
                    epoch_tqdm.set_description(f'loss: {loss.item() * args.accumulate_grad_batches}')
                    
            global_step += 1

        if accelerator.is_main_process:
            logger.info("wait for everyone")
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(f"/data2/checkpoint/{args.run_name}/epoch-{epoch + 1}")

        accelerator.wait_for_everyone()
    