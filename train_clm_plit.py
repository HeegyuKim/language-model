import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from tokenizers import Tokenizer

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from typing import List, Dict, Any
from omegaconf import OmegaConf
import fire
# from dataset import PLMDatasetForTPU
from datasets import disable_caching, load_dataset

disable_caching()



class CLMPretrainingModule(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()
        self.args = args
        self.config = AutoConfig.from_pretrained(args.model_name)
        self.model = AutoModelForCausalLM.from_config(self.config)
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.args.learning_rate)

    def training_step(self, batch, batch_idx):
        ids = batch["input_ids"]
        loss = self.model(input_ids=ids, labels=ids).loss
        return loss
    

def main(config_name: str = "test.yaml"):
    args = OmegaConf.load(f"config/{config_name}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    dataset = load_dataset("json", data_dir=args.data_dir, split="train").with_format("torch")
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    print("data total", len(dataset), "blocks")

    module = CLMPretrainingModule(args)

    save_dir = f"./checkpoint/{args.run_name}"
    checkpoint_callback = ModelCheckpoint(
        dirpath=save_dir,
        filename="{step}",
        save_last=True,
        every_n_train_steps=args.save_steps,
        save_top_k=-1
    )

    callbacks = [
        checkpoint_callback,
    ]

    trainer = pl.Trainer(
        max_epochs=1,
        callbacks=callbacks,
        accelerator=args.accelerator,
        devices=args.devices
        )
    
    trainer.fit(module, dataloader)
            
if __name__ == "__main__":
    fire.Fire(main)