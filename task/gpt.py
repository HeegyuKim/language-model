from typing import List, Dict
from .base import BaseTask

import torch
import evaluate
# from korouge_score import rouge_scorer
import re
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
            dataset = self.dataset.map(
                self._encode_data, remove_columns=self.dataset["train"].column_names
            ).filter(self._filter_instance, load_from_cache_file=True)

            self.mapped_dataset = dataset

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
    
    def _filter_instance(self, x):
        return True

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

class LyricsGPTTask(CausalFineTuningTask):
    def prepare_dataset(self):
        self.dataset = load_dataset("heegyu/bugs-lyrics", split="train").train_test_split(0.1)
        
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
        title = x["title"]
        artist = x["artist"]
        text = f"제목: {title}\n아티스트: {artist}\n가사:\n{x['lyrics'].strip()}" + self.tokenizer.eos_token
        ids = self.tokenizer.encode(
            text, truncation=True, max_length=self.model_args.max_sequence_length
        )
        out = {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": ids}
        return out


class GoraniTask(CausalFineTuningTask):
    def prepare_dataset(self):
        self.dataset = load_dataset("heegyu/open-korean-instructions", split="train", revision='singleturn').train_test_split(0.05)
        
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
        usr, bot, sys = '<usr>','<bot>','<sys>'
        all_ids, all_labels = [], []
        speaker_tokens = [r'<usr>',r'<bot>',r'<sys>']
        split_points = [m.start(0) for r in speaker_tokens for m in re.finditer(r, x["text"])]
        split_points.sort()
        split_points.append(-1)

        max_length = self.model_args.max_sequence_length
        for i in range(len(split_points) - 1):
            begin, end = split_points[i], split_points[i + 1]
            uttr = x['text'][begin:end].strip()
            if len(uttr) == 0:
                continue
            
            ids = self.tokenizer.encode(
                uttr, truncation=True, max_length=max_length
            )

            if uttr.startswith(bot):
                ids.append(self.tokenizer.eos_token_id)
                labels = ids
            else:
                labels = [-100] * len(ids)
            
            all_ids.extend(ids)
            all_labels.extend(labels)

            # print(uttr)
            # print(tokenizer.decode(ids))

        if len(all_ids) > max_length:
            all_ids = all_ids[:max_length]
        if len(all_labels) > max_length:
            all_labels = all_labels[:max_length]

        # if len(all_ids) > max_length:
        # print(len(all_ids), len(all_labels))
        out = {"input_ids": all_ids, "attention_mask": [1] * len(all_ids), "labels": all_labels}
        return out

    def _filter_instance(self, x):
        labels = x['labels']
        loss_labels = [l for l in labels if l != -100]

        return len(loss_labels) > 0