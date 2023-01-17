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


class AccelerateTrainer:

    def __init__(self):
        self.accelerator = Accelerator()
        self.train_dataloader = None
        self.validation_dataloader = None

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        pass

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        pass

    def train(
        self,
        max_epochs: Optional[int] = None,
        max_steps: Optional[int] = None,
        ):
        assert self.train_dataloader is not None

        self.step = 0

        for epoch in tqdm(range(args.num_epochs), position=0, desc="epoch"):
            model.train()

            for batch_idx, batch in enumerate(
                tqdm(
                    self.train_dataloader, 
                    disable=not accelerator.is_local_main_process, 
                    position=1, 
                    leave=False
                )
            ):
                loss = self.training_step(batch, batch_idx)
                
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

                self.step += 1