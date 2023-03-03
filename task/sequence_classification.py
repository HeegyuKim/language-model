from typing import List, Dict
from .base import BaseTask

import torch
import evaluate

from pprint import pprint
from datasets import load_dataset
from utils.collator import SequenceClassificationCollator
from utils.metric import ConfiguredMetric

NEWS_CATEGORIES = [
    "ENTERTAINMENT", "POLITICS", "WELLNESS", "TRAVEL", "STYLE & BEAUTY",
    "PARENTING", "HEALTHY LIVING", "QUEER VOICES", "FOOD & DRINK", "BUSINESS"
    ]
NEWS_CATEGORIES2id = {k: i for i, k in enumerate(NEWS_CATEGORIES)}

def map_news_category(x):
    x["text"] = x['short_description']
    x["label"] = NEWS_CATEGORIES2id[x["category"]]
    return x


class NewsCategoryClassificationTask(BaseTask):
    
    def prepare_dataset(self):
        self.dataset = load_dataset("heegyu/news-category-balanced-top10", split="train").train_test_split(0.1, seed=42)
        self.evaluator = evaluate.combine([
            evaluate.load("accuracy"),
            ConfiguredMetric(evaluate.load("f1"), average="macro")
        ])
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

        with self.accelerator.local_main_process_first():
            self.dataset = self.dataset.map(
                map_news_category, remove_columns=self.dataset["train"].column_names
            )

            self.mapped_dataset = self.dataset.map(
                self._encode_data, remove_columns=self.dataset["train"].column_names, 
            )

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
