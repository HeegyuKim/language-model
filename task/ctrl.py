from .base import BaseTask

import torch
import evaluate
import wandb

from pprint import pprint
from datasets import load_dataset
import numpy as np

import random
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from random import randint
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

from utils.collator import DataCollatorForCausalLM

NEWS_CATEGORIES = [
    "ENTERTAINMENT", "POLITICS", "WELLNESS", "TRAVEL", "STYLE & BEAUTY",
    "PARENTING", "HEALTHY LIVING", "QUEER VOICES", "FOOD & DRINK", "BUSINESS"
    ]
NEWS_CATEGORIES2id = {k: i for i, k in enumerate(NEWS_CATEGORIES)}

BBC_NEWS = [
    "tech", "business", "sport", "entertainment", "politics"
]
BBC_NEWS_CATEGORIES2id = {k: i for i, k in enumerate(BBC_NEWS)}

CLASS_LABELS = {
    "emotion": ["sadness", "joy", "love", "anger", "fear", "surprise"],
    "imdb": ["negative", "positive"],
    "yelp_polarity": ["negative", "positive"],
    "heegyu/news-category-balanced-top10": NEWS_CATEGORIES,
    "SetFit/bbc-news": BBC_NEWS,
}

def map_news_category(x):
    headline = x["headline"]
    desc = x["short_description"]
    x["text"] = f"Title: {headline}\nContent: {desc}"
    x["label"] = NEWS_CATEGORIES2id[x["category"]]
    return x

class CTRLTask(BaseTask):
    
    def prepare_dataset(self):
        dataset_name = self.data_args.dataset_name
        self.ctrl_labels = CLASS_LABELS[dataset_name]
        self.dataset = load_dataset(dataset_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        with self.accelerator.local_main_process_first():
            if dataset_name == "heegyu/news-category-balanced-top10":
                self.dataset = self.dataset.map(
                    map_news_category, remove_columns=self.dataset["train"].column_names
                )['train'].train_test_split(0.1, seed=42)

            self.mapped_dataset = self.dataset.map(
                self._encode_data, remove_columns=self.dataset["train"].column_names
            )

        return {
            'train': self.mapped_dataset['train'],
            'validation': self.mapped_dataset['test'],
        }

    def _encode_data(self, x):
        ctrl_code = self.ctrl_labels[x['label']]
        text = f"topic: {ctrl_code}\n" + x["text"].strip().replace("\\xa0", '') + self.tokenizer.eos_token
        ids = self.tokenizer.encode(
            text, truncation=True, max_length=self.model_args.max_sequence_length
        )
        out = {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": ids}
        return out
        

    def get_collator(self):
        return DataCollatorForCausalLM(
            tokenizer=self.tokenizer,
            max_length=self.model_args.max_sequence_length,
            pad_to_multiple_of=8,
            padding="max_length",
            return_tensors="pt"
        )

    def training_step(self, batch):
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
        all_sequences = []

        for label in self.ctrl_labels:
            prompt = f"topic: {label}\n"
            prompt = self.tokenizer.encode(prompt, return_tensors="pt")

            sequences = model.generate(prompt, max_new_tokens=64, do_sample=True, num_return_sequences=5, early_stopping=True)
            sequences = self.tokenizer.batch_decode(sequences, skip_special_tokens=True)

            for s in sequences:
                all_sequences.append((label, s))

            print(f"test generations for {label}")
            pprint(sequences)

        if wandb.run is not None:
            table = wandb.Table(['class', 'text'])
            for seq in all_sequences:
                table.add_data(*seq)

            wandb.log({'sample_generations': table})

        model.to(device)

