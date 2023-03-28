from typing import List, Dict
from .base import BaseTask

import torch
import evaluate
from korouge.korouge_score import rouge_scorer

from pprint import pprint
from datasets import load_dataset
from utils.collator import DataCollatorForCausalLM
from utils.metric import ConfiguredMetric
from mecab import MeCab
from collections import defaultdict


class NiaDialogTask(BaseTask):

    def prepare_dataset(self):
        self.dataset = load_dataset("heegyu/aihub_daily_conv_2022_CRF", use_auth_token=True)
        for k, v in self.dataset.items():
            self.dataset[k] = v.shuffle(seed=42).select(range(10000))
        self.rouge_scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL", "rougeLsum"])
        self.mecab = MeCab()

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

        prompt_text = x['context'].replace("\n", self.tokenizer.eos_token) + self.tokenizer.eos_token
        summary_text = x['response']
        
        self.tokenizer.truncation_side = "left"
        self.tokenizer.padding_side = "left"
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

        if self.training_args.do_eval_generate:
            generate_out = self.eval_generate_step(batch)
            out.update(generate_out)

        return out

    def split_korean(self, texts: List[str]):
        texts = [" ".join(self.mecab.morphs(x)) for x in texts]
        return texts

    def eval_generate_step(self, batch):
        device = self.device

        self.tokenizer.padding_side = "left"
        prompt = self.tokenizer(batch["prompt_text"], truncation=True, max_length=800, return_tensors="pt")
        prompt = {k: v.to(device) for k, v in prompt.items()}
        prompt_len = prompt["input_ids"].shape[1]

        prediction = self.model.generate(
            **prompt,
            max_new_tokens=100,
            min_length=prompt_len + 10,
            do_sample=True,
            early_stopping=True
        )
        prediction = self.tokenizer.batch_decode(prediction[:,prompt_len:], skip_special_tokens=True)
        
        # print(batch["prompt_text"])
        # print(batch["label_text"])
        # print(prediction)

        # TODO: Mecab
        prediction = self.split_korean(prediction)
        labels = self.split_korean(batch["label_text"])
        scores = defaultdict(list)
        for l, p in zip(labels, prediction):
            score = self.rouge_scorer.score(l, p)
            for k, v in score.items():
                scores[k].append(v)

        scores = {k: torch.tensor(v, device=device) for k, v in dict(scores).items()}
        return scores


    def collate_evaluation(self, results: Dict):
        eval_results = {
            k: torch.stack(v).mean().item()
            for k, v in results.items()
        }
        
        pprint("evaluation result")
        pprint(eval_results)
        return eval_results
