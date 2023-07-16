import transformers
from transformers import HfArgumentParser, set_seed

from utils.collator import SequenceClassificationCollator
from utils.arguments import TrainingArguments, DataTrainingArguments, ModelArguments
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from typing import List, Dict, Any
from datasets import disable_caching, load_dataset
from accelerate import Accelerator
from tqdm.auto import tqdm
from accelerate.logging import get_logger

import os
import evaluate
from pprint import pprint
from transformers import AutoModelForSequenceClassification, AutoModelForCausalLM, AutoConfig, AutoModelForTokenClassification
from .model.director import DirectorModel

MODEL_TYPES = {
    "sequence-classification": AutoModelForSequenceClassification,
    "token-classification": AutoModelForTokenClassification,
    "causal-lm": AutoModelForCausalLM,
    "director": DirectorModel
}


def collate_dictlist(dl):
    from collections import defaultdict

    out = defaultdict(list)

    for d in dl:
        for k, v in d.items():
            out[k].append(v)

    return out

class BaseTask:
    def __init__(self, accelerator, training_args, data_args, model_args) -> None:
        super().__init__()
        self.accelerator = accelerator
        self.training_args = training_args
        self.data_args = data_args
        self.model_args = model_args

    @property
    def device(self):
        return next(self.model.parameters()).device

    def get_model(self, args):
        model_cls = MODEL_TYPES[args.model_type]

        kwargs = {}

        if args.model_type == "sequence-classification":
            kwargs["num_labels"] = args.num_labels

        if args.config_name is not None:
            config = AutoConfig.from_pretrained(args.config_name)
            model = model_cls(config, **kwargs)
        elif args.model_name_or_path is not None:
            model = model_cls.from_pretrained(
                args.model_name_or_path,
                revision=args.revision,
                from_flax=args.from_flax,
                **kwargs
                )
        else:
            raise Exception("config_name or model_name_or_path 가 지정되어야 합니다.")
    
        return model

    def get_tokenizer(self, args):
        from transformers import AutoTokenizer

        if args.tokenizer_name is not None:
            return AutoTokenizer.from_pretrained(args.tokenizer_name)
        elif args.model_name_or_path is not None:
            return AutoTokenizer.from_pretrained(args.model_name_or_path)
        else:
            raise Exception("config_name or model_name_or_path 가 지정되어야 합니다.")

    def setup(self):
        self.model = self.get_model(self.model_args)
        self.tokenizer = self.get_tokenizer(self.model_args)

        datasets = self.prepare_dataset()

        self.train_dataloader = self._create_dataloader(
            datasets.get('train'),
            True
        )
        self.eval_dataloader = self._create_dataloader(
            datasets.get('validation'),
        )

        steps_per_epoch = len(datasets.get('train')) / (
            self.training_args.per_device_train_batch_size
            * self.training_args.gradient_accumulation_steps
        )
        total_steps = int(self.training_args.num_train_epochs * steps_per_epoch)
        optimizer = optim.AdamW(self.model.parameters(), lr=self.training_args.learning_rate)

        # lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        #     optimizer,
        #     max_lr=self.training_args.learning_rate,
        #     steps_per_epoch=steps_per_epoch,
        #     epochs=self.training_args.num_train_epochs,
        #     anneal_strategy="linear",
        #     pct_start=0.01,
        #     div_factor=10,
        #     final_div_factor=10,
        # )
        lr_scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer, 
            total_steps // 20, # 0.05 
            total_steps
            )

        (
            self.model,
            self.optimizer,
            self.train_dataloader,
            self.lr_scheduler,
            self.eval_dataloader,
        ) = self.accelerator.prepare(
            self.model, optimizer, self.train_dataloader, lr_scheduler, self.eval_dataloader
        )

        self.accelerator.register_for_checkpointing(lr_scheduler)

    def prepare_dataset(self):
        pass

    def get_collator(self):
        return None

    def training_step(self, batch):
        """
            return loss
        """
        pass

    def evaluation_step(self, batch):
        """
            return dict
        """
        pass

    def collate_evaluation(self, results: List[Dict]):
        """
            return dict(metric)
        """
        return None

    def _create_dataloader(self, dataset, shuffle=True):
        if dataset is None:
            return None

        kwargs = {}
        collator = self.get_collator()
        if collator is not None:
            kwargs['collate_fn'] = collator

        return DataLoader(
            dataset,
            batch_size=self.training_args.per_device_train_batch_size,
            shuffle=shuffle,
            **kwargs
            )

    def train(self):
        global_step = 0
        optimizer_step = 0

        for epoch in tqdm(
            range(self.training_args.num_train_epochs),
            position=0,
            disable=not self.accelerator.is_local_main_process,
        ):
            self.model.train()
            epoch_tqdm = tqdm(
                self.train_dataloader,
                disable=not self.accelerator.is_local_main_process,
                position=1,
                leave=False,
            )

            for step, batch in enumerate(epoch_tqdm):
                step_output = self.training_step(batch)
                if torch.is_tensor(step_output):
                    loss = step_output
                else:
                    loss = step_output['loss']
                loss = loss / self.training_args.gradient_accumulation_steps
                self.accelerator.backward(loss)

                if (global_step + 1) % self.training_args.gradient_accumulation_steps == 0:
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)

                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                    optimizer_step += 1

                    if (
                        self.accelerator.is_main_process
                        and optimizer_step % self.training_args.logging_steps == 0
                    ):  
                        if torch.is_tensor(step_output):
                            metrics = {"train/loss": step_output.item()}
                        else:
                            metrics = {f"train/{k}": v.item() for k, v in step_output.items()}

                        metrics["optimizer_step"] = optimizer_step
                        metrics["train/learning_rate"] = self.lr_scheduler.scheduler._last_lr[0]
                        metrics["train/loss"] = loss.item() * self.training_args.gradient_accumulation_steps
                        self.accelerator.log(metrics)

                    epoch_tqdm.set_description(
                        f"loss: {loss.item() * self.training_args.gradient_accumulation_steps}"
                    )

                    if (
                        self.training_args.do_eval
                        and self.training_args.eval_strategy == "steps"
                        and optimizer_step % self.training_args.eval_steps == 0
                    ):
                        self.evaluate(epoch, optimizer_step)

                global_step += 1

            if self.training_args.save_strategy == "epoch":
                if self.accelerator.is_main_process:
                    self.save_model(f"epoch-{epoch}")
                self.accelerator.wait_for_everyone()

            if self.training_args.do_eval and self.training_args.eval_strategy == "epoch":
                self.evaluate(epoch, optimizer_step)

        if self.training_args.save_strategy == "last":
            if self.accelerator.is_main_process:
                self.save_model(f"epoch-{epoch}-last")
            self.accelerator.wait_for_everyone()

    def save_model(self, name):
        run_name = self.training_args.run_name.replace("/", "__")
        path = f"{self.training_args.output_dir}/{run_name}/{name}"
        device = next(self.model.parameters()).device
        unwrapped_model = self.accelerator.unwrap_model(self.model).cpu()
        unwrapped_model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

        unwrapped_model.to(device)

    @torch.no_grad()
    def evaluate(self, epoch, optimizer_step):
        self.model.eval()

        epoch_tqdm = tqdm(
            self.eval_dataloader,
            disable=not self.accelerator.is_local_main_process,
            position=1,
            leave=False,
        )
        step_outputs = []
        for step, batch in enumerate(epoch_tqdm):
            outputs = self.evaluation_step(batch)
            if torch.is_tensor(outputs):
                outputs = {"loss": outputs}
            step_outputs.append(outputs)

        eval_outputs = self.accelerator.gather_for_metrics(step_outputs)

        if self.accelerator.is_local_main_process:
            eval_outputs = collate_dictlist(eval_outputs)
            eval_results = self.collate_evaluation(eval_outputs)
            eval_results = {f"eval/{k}": v for k, v in eval_results.items()}
            self.accelerator.log(eval_results)

        self.accelerator.wait_for_everyone()
        self.model.train()
