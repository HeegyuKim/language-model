from typing import List, Dict
from .base import BaseTask

import torch
import evaluate
# from korouge_score import rouge_scorer

from pprint import pprint
from datasets import load_dataset
from utils.metric import ConfiguredMetric

from utils.collator import DataCollatorForCausalLM


class GPTTask(BaseTask):

    def prepare_dataset(self):
        self.dataset = load_dataset(
            "json",
            data_files={
                "train": self.data_args.train_file,
                "validation": self.data_args.validation_file
            }
        ).with_format("torch")
        return self.dataset


    def training_step(self, batch):
        return self.model(
            input_ids=batch["input_ids"],
            labels=batch["input_ids"],
        ).loss

    def evaluation_step(self, batch):
        return self.model(
            input_ids=batch["input_ids"],
            labels=batch["input_ids"],
        ).loss

    def collate_evaluation(self, results: Dict):
        eval_results = {
            k: torch.stack(v).mean().item()
            for k, v in results.items()
        }
        
        pprint("evaluation result")
        pprint(eval_results)
        return eval_results


class CausalFineTuningTask(BaseTask):
    
    def prepare_dataset(self):
        dataset_name = self.data_args.dataset_name
        self.dataset = load_dataset(dataset_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        with self.accelerator.local_main_process_first():
            self.mapped_dataset = self.dataset.map(
                self._encode_data, remove_columns=self.dataset["train"].column_names
            )

        return {
            'train': self.mapped_dataset['train'],
            'validation': self.mapped_dataset['validation'],
        }

    def _encode_data(self, x):
        text = self.tokenizer.bos_token + x["content"].strip().replace("\\xa0", '') + self.tokenizer.eos_token
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

        return eval_results
