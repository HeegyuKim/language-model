from .base import BaseTask

import torch
import evaluate

from pprint import pprint
from datasets import load_dataset
import numpy as np

import random
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from random import randint
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

from transformers import DataCollatorForSeq2Seq


CLASS_LABELS = {
    "emotion": ["sadness", "joy", "love", "anger", "fear", "surprise"],
    "imdb": ["negative", "positive"],
    "yelp_polarity": ["negative", "positive"],
}


class CTRLTask(BaseTask):
    
    def prepare_dataset(self):
        dataset_name = self.data_args.dataset_name
        self.ctrl_labels = CLASS_LABELS[dataset_name]
        self.dataset = load_dataset(dataset_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        with self.accelerator.local_main_process_first():
            self.mapped_dataset = self.dataset.map(
                self._encode_data, remove_columns=self.dataset["train"].column_names
            )

        return {
            'train': self.mapped_dataset['train'],
            'validation': self.mapped_dataset['test'],
        }

    def _encode_data(self, x):
        ctrl_code = self.ctrl_labels[x['label']]
        text = f"{ctrl_code} " + x["text"]
        ids = self.tokenizer.encode(
            text, truncation=True, max_length=self.model_args.max_sequence_length
        )
        out = {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": ids}
        return out
        

    def get_collator(self):
        return DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            max_length=self.model_args.max_sequence_length,
            pad_to_multiple_of=8,
            padding="max_length",
            return_tensors="pt"
        )

    def training_step(self, batch):
        for k, v in batch.items():
            print(k, v.shape)
        out = self.model(**batch)

        return {
            'loss': out.loss,
        }

    def evaluation_step(self, batch):
        out = self.model(**batch)

        return {
            'loss': out.loss,
        }

    def collate_evaluation(self, results: List[Dict]):
        eval_mean_loss = torch.stack(results['loss']).mean().item()
        eval_results = {
            "loss": eval_mean_loss,
        }
        pprint("evaluation result")
        pprint(eval_results)

        self.test_generation()
        return eval_results

    def test_generation(self):        
        model = self.accelerator.unwrap_model(self.model)
        device = next(model.parameters()).device
        model = model.cpu().eval()

        for label in self.ctrl_labels:
            prompt = f"{label} this is really "
            prompt = self.tokenizer.encode(prompt, return_tensors="pt")

            sequences = model.generate(prompt, max_new_tokens=32, do_sample=True, num_return_sequences=5)
            sequences = self.tokenizer.batch_decode(sequences)

            print(f"test generations for {label}")
            pprint(sequences)

        model.to(device)

