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

from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy
from transformers import AutoTokenizer, AutoModelForSequenceClassification

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
        batch['labels'] = batch['input_ids']
        return batch


class DirectorTask(BaseTask):

    def setup(self):
        super().setup()

        if self.model_args.director_frozen:
            self.model.freeze_gpt()

        with self.accelerator.local_main_process_first():
            self.toxic_tokenizer = AutoTokenizer.from_pretrained(self.model_args.director_eval_classifier)
            self.toxic_model = AutoModelForSequenceClassification.from_pretrained(self.model_args.director_eval_classifier).eval()
    
    @torch.no_grad()
    def classify_toxic(self, texts):
        encoded_input = self.toxic_tokenizer(texts, return_tensors='pt', truncation=True, padding=True, max_length=512)
        output = self.toxic_model(**encoded_input).logits.softmax(-1)[:, 1]
        # scores = output.argmax(-1)
        return output.tolist()


    def prepare_dataset(self):
        # self.dataset = load_dataset("hate_speech18", split="train").train_test_split(0.1, seed=42)
        self.dataset = load_dataset(self.data_args.dataset_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        with self.accelerator.local_main_process_first():
            self.mapped_dataset = self.dataset.map(
                self._encode_data, remove_columns=self.dataset["train"].column_names
            )

        if "validation" not in self.dataset:
            if "test" not in self.dataset:
                self.dataset = self.dataset['train'].train_test_split(0.1, seed=42)
            return {
                'train': self.mapped_dataset['train'],
                'validation': self.mapped_dataset['test'],
            }
        else:
            return self.dataset

    def _encode_data(self, x):
        ids = self.tokenizer.encode(
            x["text"], truncation=True, max_length=self.model_args.max_sequence_length
        )
        out = {"input_ids": ids, "attention_mask": [1] * len(ids)}
        out["class_labels"] = x["label"] if 'label' in x else x["class"]
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
        out = self.model(**batch, gamma=self.model_args.director_gamma_train)
        norm_loss = self.model.explicit_normalization_loss(batch['labels'], out.class_logits)

        return {
            'loss': out.loss + norm_loss,
            'class_loss': out.class_loss,
            'norm_loss': norm_loss
        }

    def evaluation_step(self, batch):
        out = self.model(**batch)
        norm_loss = self.model.explicit_normalization_loss(batch['labels'], out.class_logits)

        return {
            'loss': out.loss,
            'class_loss': out.class_loss,
            'norm_loss': norm_loss
        }

    def collate_evaluation(self, results: List[Dict]):
        eval_mean_loss = torch.stack(results['loss']).mean().item()
        eval_mean_class_loss = torch.stack(results['class_loss']).mean().item()
        eval_mean_norm_loss = torch.stack(results['norm_loss']).mean().item()
        eval_results = {
            "loss": eval_mean_loss,
            "class_loss": eval_mean_class_loss,
            "norm_loss": eval_mean_norm_loss
        }
        pprint("evaluation result")
        pprint(eval_results)

        self.test_generation()
        return eval_results

    def test_generation(self):        
        model = self.accelerator.unwrap_model(self.model)
        device = next(model.parameters()).device
        model = model.cpu().eval()

        prompt = "아니 "
        prompt = self.tokenizer.encode(prompt, return_tensors="pt")

        all_sequences = []

        for positive in [False, True]:
            sequences = model.generate(
                prompt, max_new_tokens=32, min_length=32, generate_positive=positive,
                no_repeat_ngram_size=4,
                do_sample=True, num_return_sequences=10, gamma=self.model_args.director_gamma_generate)
            sequences = self.tokenizer.batch_decode(sequences, skip_special_tokens=True)
            toxicities = self.classify_toxic(sequences)
            toxic = "toxic" if positive else "non-toxic"
            print("test generations", toxic)
            pprint(sequences)

            for seq, toxicity in zip(sequences, toxicities):
                all_sequences.append((toxic, seq, toxicity))

        if wandb.run is not None:
            table = wandb.Table(['class', 'text', 'toxicity'])
            for seq in all_sequences:
                table.add_data(*seq)

            wandb.log({'sample_generations': table})

        model.to(device)
