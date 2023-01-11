# Pytorch Training using huggingface accelerate
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from tokenizers import Tokenizer
import wandb

from typing import List, Dict, Any
from omegaconf import OmegaConf
import fire
from datasets import disable_caching, load_dataset
from accelerate import Accelerator
from tqdm.auto import tqdm
# import jax
# disable_caching()


def main():
    args = OmegaConf.load(f"config/ajoublue-gpt2-base.yaml")
    # dataset = load_dataset("json", data_dir=args.data_dir, split="train", cache_dir="/data/.cache").with_format("torch")
    dataset = load_dataset("json", data_files=["/data/v1-vocab51k-block1024/heegyu__kowikitext.jsonl"], split="train", cache_dir="/data/.cache").with_format("torch")
    print("data total", len(dataset), "blocks")
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    steps_per_epoch = len(dataset) // args.batch_size // 8

    config = AutoConfig.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_config(config)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    # lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
    #     optimizer, 
    #     max_lr=0.1,
    #     steps_per_epoch=steps_per_epoch,  
    #     epochs=args.num_epochs,
    #     anneal_strategy='linear',
    #     )

    accelerator = Accelerator()


    model, optimizer, train_dataloader  = accelerator.prepare(
        model, optimizer, dataloader
    )

    global_step = 0

    if accelerator.is_main_process:
        wandb.init(project=args.project, name=args.run_name)

    for epoch in tqdm(range(args.num_epochs), position=0):
        model.train()
        epoch_tqdm = tqdm(train_dataloader, disable=not accelerator.is_local_main_process, position=1, leave=False)
        for step, batch in enumerate(epoch_tqdm):
            ids = batch["input_ids"]
            loss = model(input_ids=ids, labels=ids).loss
            accelerator.backward(loss)

            # if accelerator.sync_gradients:
            #     accelerator.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            # lr_scheduler.step()
            optimizer.zero_grad()

            # if accelerator.is_main_process and step % args.logging_steps == 0:
            #     wandb.log({
            #         'global_step': global_step,
            #         'epoch': epoch + step / steps_per_epoch,
            #         'loss': loss.item()
            #     })
            #     epoch_tqdm.set_description(f'loss: {loss.item()}')

            global_step += 1

        if accelerator.is_main_process:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(f"/data/checkpoint/{args.run_name}/{epoch + 1}")