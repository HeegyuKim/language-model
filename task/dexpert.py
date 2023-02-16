from typing import List, Dict
from .base import BaseTask

import torch
import evaluate
from transformers import DataCollatorForLanguageModeling

from pprint import pprint
from datasets import load_dataset
from utils.metric import ConfiguredMetric
from utils.collator import DataCollatorForCausalLM


class BaseDExpertTask(BaseTask):
    def _filter_item(self, x):
        return True

    def prepare_dataset(self):
        self.dataset = load_dataset("SetFit/toxic_conversations")

        # for k, v in self.dataset.items():
        #     if len(self.dataset[k]) >= 1000:
        #         self.dataset[k] = self.dataset[k].select(range(1000))
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

        with self.accelerator.local_main_process_first():
            self.dataset = self.dataset.filter(self._filter_item)
            self.mapped_dataset = self.dataset.map(
                self._encode_data, remove_columns=self.dataset["train"].column_names, 
            )

            print(self.mapped_dataset)

        return {
            "train": self.mapped_dataset["train"],
            "validation": self.mapped_dataset["test"]
        }

    def _encode_data(self, x):
        ids = self.tokenizer.encode(
            x["text"], truncation=True, max_length=self.model_args.max_sequence_length
        )
        out = {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": ids}
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
        return self.model(**batch).loss

    def evaluation_step(self, batch):
        out = self.model(**batch)

        return {
            'loss': out.loss
        }

    def collate_evaluation(self, results: List[Dict]):
        eval_mean_loss = torch.stack(results['loss']).mean().item()
        eval_results = {
            "loss": eval_mean_loss,
        }
        pprint("evaluation result")
        pprint(eval_results)
        return eval_results

class ToxicDExpertTask(BaseDExpertTask):

    def _filter_item(self, x):
        return x['label'] >= 0.5

class NonToxicDExpertTask(BaseDExpertTask):
    def _filter_item(self, x):
        return x['label'] < 0.5