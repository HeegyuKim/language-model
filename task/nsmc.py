from typing import List, Dict
from .base import BaseTask

import torch
import evaluate

from pprint import pprint
from datasets import load_dataset
from utils.collator import SequenceClassificationCollator

class NSMCTask(BaseTask):
    
    def prepare_dataset(self):
        self.dataset = load_dataset("nsmc")
        self.eval_accuracy = evaluate.load("accuracy")
        
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
            x["document"], truncation=True, max_length=self.model_args.max_sequence_length
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
            "accuracy": self.eval_accuracy.compute(predictions=preds, references=labels)['accuracy']
        }
        pprint("evaluation result")
        pprint(eval_results)
        return eval_results
