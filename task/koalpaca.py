from typing import List, Dict
from .base import BaseTask

import torch

from pprint import pprint
from datasets import load_dataset
from utils.collator import DataCollatorForCausalLM
from utils.metric import ConfiguredMetric
from collections import defaultdict


class KoAlpacaTask(BaseTask):

    def prepare_dataset(self):
        self.dataset = load_dataset("Bingsu/ko_alpaca_data", split="train") \
                            .train_test_split(0.1)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

        with self.accelerator.local_main_process_first():
            self.mapped_dataset = self.dataset.map(
                self._encode_data, remove_columns=self.dataset["train"].column_names, 
            )

        return {
            "train": self.mapped_dataset["train"],
            "validation": self.mapped_dataset["test"]
        } 

    def _encode_data(self, x):
        max_length = self.model_args.max_sequence_length
        
        
        prompt_text = f"<usr>{x['instruction']}\n"
        if len(x['input']) > 0:
            prompt_text += x['input'] + "\n"
        summary_text = "<sys>" + x['output']
        
        prompt = self.tokenizer.encode(prompt_text, truncation=True, max_length=max_length)
        summary = self.tokenizer.encode(summary_text + self.tokenizer.eos_token, truncation=True, max_length=max_length)

        if len(prompt) + len(summary) > max_length:
            prompt = prompt[:max_length - len(summary)]
        
        ids = prompt + summary
        labels = [-100] * len(prompt) + summary

        out = {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}
        out["prompt_text"] = prompt_text
        out["label_text"] = summary_text
        return out
        

    def get_collator(self):
        return DataCollatorForCausalLM(
            tokenizer=self.tokenizer,
            max_length=self.model_args.max_sequence_length,
            pad_to_multiple_of=8,
            padding="max_length",
            return_tensors="pt",
        )

    def training_step(self, batch):
        return self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        ).loss

    def evaluation_step(self, batch):
        loss = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        ).loss

        out = {
            'loss': loss
        }

        return out


    def collate_evaluation(self, results: Dict):
        eval_results = {
            k: torch.stack(v).mean().item()
            for k, v in results.items()
        }
        
        pprint("evaluation result")
        pprint(eval_results)
        return eval_results
