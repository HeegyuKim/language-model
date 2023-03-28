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

from utils.collator import DataCollatorForCausalLM, SequenceClassificationCollator


def segments_intersect(x1, x2, y1, y2):
    # Assumes x1 <= x2 and y1 <= y2; if this assumption is not safe, the code
    # can be changed to have x1 being min(x1, x2) and x2 being max(x1, x2) and
    # similarly for the ys.
    return x2 >= y1 and y2 >= x1

def offset_in_positions(a, b, probs):
    for pa, pb in probs:
        if segments_intersect(a, b, pa, pb):
            return 1
    return 0

def make_labels(item, offset_mapping):
    probs = eval(item['probability'])
    probs = [k for k, v in probs.items() if v >= 0.5]
    token_labels = [offset_in_positions(a, b, probs) for a, b in offset_mapping]
    return token_labels


class ToxicSpanDetectionTask(BaseTask):
    
    def prepare_dataset(self):
        self.loss_fct = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, 100.0]))
        self.dataset = load_dataset("heegyu/toxic-spans")
        self.evaluator = evaluate.combine([
            evaluate.load("accuracy"),
            evaluate.load("f1")
        ])
        
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

    def _encode_data(self, item):
        text = item["text"].strip()
        out = self.tokenizer(
            text, 
            truncation=True, 
            max_length=self.model_args.max_sequence_length,
            return_offsets_mapping=True
        )

        out["labels"] = make_labels(item, out['offset_mapping'])
        del out["offset_mapping"]

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
        labels = batch.pop("labels")
        logits = self.model(**batch).logits

        return {
            'loss': self.compute_loss(logits, labels)
        }

    def evaluation_step(self, batch):
        labels = batch.pop("labels")
        logits = self.model(**batch).logits

        return {
            'loss': self.compute_loss(logits, labels),
            'logits': logits,
            'labels': labels
        }

    def compute_loss(self, logits, labels):
        loss = self.loss_fct(logits.view(-1, 2), labels.view(-1))
        return loss
        

    def collate_evaluation(self, results: List[Dict]):
        logits = torch.stack(results['logits'])
        preds = logits.view(-1, logits.shape[-1]).argmax(-1)
        labels = torch.stack(results['labels']).view(-1)
        print(preds, labels)
        eval_mean_loss = torch.stack(results['loss']).mean().item()

        labels_ok = labels >= 0
        preds, labels = preds[labels_ok], labels[labels_ok]

        eval_results = {
            "loss": eval_mean_loss,
            **self.evaluator.compute(predictions=preds, references=labels)
        }
        pprint("evaluation result")
        pprint(eval_results)
        return eval_results



class ToxicSequenceClassificationTask(BaseTask):
    
    def prepare_dataset(self):
        self.dataset = load_dataset("heegyu/toxic_conversations_balanced")
        self.evaluator = evaluate.combine([
            evaluate.load("accuracy"),
            evaluate.load("f1")
        ])
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

        with self.accelerator.local_main_process_first():
            self.mapped_dataset = self.dataset.map(
                self._encode_data, remove_columns=self.dataset["train"].column_names
            )

        print(self.mapped_dataset)

        return {
            'train': self.mapped_dataset['train'],
            'validation': self.mapped_dataset['test'],
        }

    def _encode_data(self, x):
        ids = self.tokenizer.encode(
            x["text"], truncation=True, max_length=self.model_args.max_sequence_length
        )
        out = {"input_ids": ids, "attention_mask": [1] * len(ids)}
        out["label"] = x["label"]
        return out
        

    def get_collator(self):
        return SequenceClassificationCollator(
            tokenizer=self.tokenizer,
            max_length=self.model_args.max_sequence_length,
            pad_to_multiple_of=8,
            padding="max_length",
            return_tensors="pt",
        )

    def training_step(self, batch):
        return self.model(**batch).loss

    def evaluation_step(self, batch):
        out = self.model(**batch)
        loss, logits = out.loss, out.logits

        return {
            'loss': loss,
            'logits': logits,
            'labels': batch['labels']
        }

    def collate_evaluation(self, results: List[Dict]):
        preds = torch.stack(results['logits']).view(-1, 2).argmax(-1)
        labels = torch.stack(results['labels']).view(-1)
        eval_mean_loss = torch.stack(results['loss']).mean().item()
        eval_results = {
            "loss": eval_mean_loss,
            **self.evaluator.compute(predictions=preds, references=labels)
        }
        pprint("evaluation result")
        pprint(eval_results)
        return eval_results
