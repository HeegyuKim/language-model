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

# disable_caching()


def main():
    args = OmegaConf.load(f"config/test.yaml")
    dataset = load_dataset("json", data_dir=args.data_dir, split="train").with_format("torch")
    print("data total", len(dataset), "blocks")
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    config = AutoConfig.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_config(config)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    accelerator = Accelerator()


    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, dataloader
    )

    for epoch in range(args.num_epochs):
        model.train()
        for step, batch in enumerate(tqdm(train_dataloader, disable=not accelerator.is_local_main_process)):
            ids = batch["input_ids"]
            loss = model(input_ids=ids, labels=ids).loss
            accelerator.backward(loss)

            optimizer.step()
            optimizer.zero_grad()

            # if accelerator.is_local_main_process and (step + 1) % 10 == 0:
            #     print('step', step)
        