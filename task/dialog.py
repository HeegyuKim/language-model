from typing import List, Dict
from .base import BaseTask

import torch
import evaluate
# from korouge_score import rouge_scorer

from pprint import pprint
from datasets import load_dataset
from utils.collator import DataCollatorForCausalLM
from utils.metric import ConfiguredMetric


class NiaSummarizationTask(BaseTask):

    def prepare_dataset(self):
        self.dataset = load_dataset("heegyu/aihub_daily_conv_2022_CRF", use_auth_token=True)
        # self.rouge_scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL", "rougeLsum"])
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

        with self.accelerator.local_main_process_first():
            self.mapped_dataset = self.dataset.map(
                self._encode_data, remove_columns=self.dataset["train"].column_names, 
            )

        return self.mapped_dataset

    def _encode_data(self, x):
        max_length = self.model_args.max_sequence_length

        prompt_text = f"{x['context']}\n"
        label_text = x['response']
        
        self.tokenizer.truncation_side = "left"
        prompt = self.tokenizer.encode(prompt_text, truncation=True, max_length=max_length)
        self.tokenizer.truncation_side = "right"
        label = self.tokenizer.encode(label_text, truncation=True, max_length=max_length)

        if len(prompt) + len(label) > max_length:
            prompt = prompt[-(max_length - len(label)):]
        
        ids = prompt + label
        labels = [-100] * len(prompt) + label

        out = {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}
        out["prompt_text"] = prompt_text
        out["label_text"] = label_text
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

    def eval_generate_step(self, batch):
        device = self.device

        self.tokenizer.padding_side = "left"
        prompt = self.tokenizer(batch["prompt_text"], truncation=True, max_length=800)
        prompt = {k: v.to(device) for k, v in prompt.items()}

        prediction = self.model.generate(
            **prompt,
            max_new_tokens=100,
            num_beams=4,
            do_sample=False
        )
        prediction = self.tokenizer.batch_decode(prediction)

        # TODO: Mecab

        # scores = self.rouge_scorer.score(batch["label_text"], prediction)
        # scores = {k: torch.tensor(v.fmeasure, device=device) for k, v in scores.items()}
        return scores


    def collate_evaluation(self, results: Dict):
        eval_results = {
            k: torch.stack(v).mean().item()
            for k, v in results.items()
        }
        
        pprint("evaluation result")
        pprint(eval_results)
        return eval_results
