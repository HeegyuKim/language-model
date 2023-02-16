from typing import List, Dict
from .base import BaseTask

import torch
import evaluate

from pprint import pprint
from datasets import load_dataset
from transformers import DataCollatorForSeq2Seq
from utils.metric import ConfiguredMetric


class NiaSummarizationTask(BaseTask):

    def prepare_dataset(self):
        self.dataset = load_dataset("heegyu/nia_summary")
        
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
        max_length = self.model_args.max_sequence_length

        prompt = f"내용: {x['passage']}\n요약: "
        summary = x['summary']
        
        prompt = self.tokenizer.encode(prompt, truncation=True, max_length=max_length)
        summary = self.tokenizer.encode(summary, truncation=True, max_length=max_length)

        if len(prompt) + len(summary) > max_length:
            prompt = prompt[:max_length - len(summary)]
        
        labels = [-100] * len(prompt) + summary

        out = {"input_ids": ids, "attention_mask": [1] * len(ids), "labels":labels}
        
        return out
        

    def get_collator(self):
        return DataCollatorForSeq2Seq(
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
        loss = out.loss

        model = self.accelerator.unwrap_model(self.model)

        return {
            'loss': loss,
            'logits': logits,
            'labels': batch['labels']
        }

    def collate_evaluation(self, results: List[Dict]):
        eval_mean_loss = torch.stack(results['loss']).mean().item()
        eval_results = {
            "loss": eval_mean_loss
        }
        pprint("evaluation result")
        pprint(eval_results)
        return eval_results
