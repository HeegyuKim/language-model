from typing import List, Dict
from .base import BaseTask

import torch
import evaluate
# from korouge_score import rouge_scorer

from pprint import pprint
from datasets import load_dataset
from utils.metric import ConfiguredMetric


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