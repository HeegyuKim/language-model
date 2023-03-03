from typing import List, Dict
from .base import BaseTask

import torch
import evaluate

from pprint import pprint
from datasets import load_dataset
from utils.collator import SequenceClassificationCollator
from utils.metric import ConfiguredMetric


class YNATTask(BaseTask):
    
    def prepare_dataset(self):
        self.dataset = load_dataset("klue", "ynat")
        self.evaluator = evaluate.combine([
            evaluate.load("accuracy"),
            ConfiguredMetric(evaluate.load("f1"), average="macro")
        ])
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

        with self.accelerator.local_main_process_first():
            self.mapped_dataset = self.dataset.map(
                self._encode_data, remove_columns=self.dataset["train"].column_names, 
            )

        print(self.mapped_dataset)

        return self.mapped_dataset

    def _encode_data(self, x):
        ids = self.tokenizer.encode(
            x["title"], truncation=True, max_length=self.model_args.max_sequence_length
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
        logits = torch.stack(results['logits'])
        preds = logits.view(-1, logits.shape[-1]).argmax(-1)
        labels = torch.stack(results['labels']).view(-1)
        eval_mean_loss = torch.stack(results['loss']).mean().item()
        eval_results = {
            "loss": eval_mean_loss,
            **self.evaluator.compute(predictions=preds, references=labels, average='macro')
        }
        pprint("evaluation result")
        pprint(eval_results)
        return eval_results


class STSBinaryTask(BaseTask):
    
    def prepare_dataset(self):
        self.dataset = load_dataset("klue", "sts")
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
                self._encode_data, remove_columns=self.dataset["train"].column_names, 
            )

        print(self.mapped_dataset)

        return self.mapped_dataset

    def _encode_data(self, x):
        s1, s2 = x['sentence1'], x['sentence2']
        text = f"1: {s1}\n2:{s2}"
        ids = self.tokenizer.encode(
            text, truncation=True, max_length=self.model_args.max_sequence_length
        )
        out = {"input_ids": ids, "attention_mask": [1] * len(ids)}
        out["label"] = x["labels"]['binary-label']
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
        logits = torch.stack(results['logits'])
        preds = logits.view(-1, logits.shape[-1]).argmax(-1)
        labels = torch.stack(results['labels']).view(-1)
        eval_mean_loss = torch.stack(results['loss']).mean().item()
        eval_results = {
            "loss": eval_mean_loss,
            **self.evaluator.compute(predictions=preds, references=labels)
        }
        pprint("evaluation result")
        pprint(eval_results)
        return eval_results
