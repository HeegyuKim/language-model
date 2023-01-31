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

from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy

@dataclass
class DirectorCollator(object):
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"


    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        class_labels = [x['class_labels'] for x in features]
        if self.return_tensors == 'pt':
            class_labels = torch.tensor(class_labels, dtype=torch.long)

        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        
        del batch['class_labels']
        batch['class_labels'] = class_labels

        return batch


class DirectorTask(BaseTask):
    
    def prepare_dataset(self):
        self.dataset = load_dataset("hate_speech18").train_test_split(0.1)
        
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
        ids = self.tokenizer.encode(
            x["text"], truncation=True, max_length=self.model_args.max_sequence_length
        )
        out = {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": ids}
        out["class_labels"] = x["label"]
        return out
        

    def get_collator(self):
        return DirectorCollator(
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

        return {
            'loss': out.loss,
            'class_loss': out.class_loss
        }

    def collate_evaluation(self, results: List[Dict]):
        eval_mean_loss = torch.stack(results['loss']).mean().item()
        eval_results = {
            "loss": eval_mean_loss,
        }
        pprint("evaluation result")
        pprint(eval_results)
        return eval_results



