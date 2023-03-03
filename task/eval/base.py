from transformers import HfArgumentParser, set_seed
from utils.collator import SequenceClassificationCollator
from utils.arguments import TrainingArguments, DataTrainingArguments, ModelArguments
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from typing import List, Dict, Any
from datasets import disable_caching, load_dataset
from accelerate import Accelerator
from tqdm.auto import tqdm
from accelerate.logging import get_logger

import os
import evaluate
from pprint import pprint
from transformers import AutoModelForSequenceClassification, AutoModelForCausalLM, AutoConfig
from .model.director import DirectorModel
from utils.collator import DataCollatorForCausalLM
from pprint import pprint


class EvaluationTask:

    def __init__(self, accelerator, training_args, data_args, model_args) -> None:
        super().__init__()
        self.accelerator = accelerator
        self.training_args = training_args
        self.data_args = data_args
        self.model_args = model_args

    def get_model(self, args):
        model_cls = MODEL_TYPES[args.model_type]

        kwargs = {}

        if args.model_type == "sequence-classification":
            kwargs["num_labels"] = args.num_labels

        if args.config_name is not None:
            config = AutoConfig.from_pretrained(args.config_name)
            model = model_cls(config, **kwargs)
        elif args.model_name_or_path is not None:
            model = model_cls.from_pretrained(
                args.model_name_or_path,
                revision=args.revision,
                from_flax=args.from_flax,
                **kwargs
                )
        else:
            raise Exception("config_name or model_name_or_path 가 지정되어야 합니다.")
    
        return model

    def get_tokenizer(self, args):
        from transformers import AutoTokenizer

        if args.tokenizer_name is not None:
            return AutoTokenizer.from_pretrained(args.tokenizer_name)
        elif args.model_name_or_path is not None:
            return AutoTokenizer.from_pretrained(args.model_name_or_path)
        else:
            raise Exception("config_name or model_name_or_path 가 지정되어야 합니다.")

    def setup(self):
        self.model = self.get_model(self.model_args)
        self.tokenizer = self.get_tokenizer(self.model_args)

        if not self.model.config.is_encoder_decoder:
            self.tokenizer.padding_side = "left"

        datasets = self.prepare_dataset()

        self.train_dataloader = self._create_dataloader(
            datasets.get('train'),
            True
        )
        self.eval_dataloader = self._create_dataloader(
            datasets.get('validation'),
        )

    def prepare_dataset(self):
        pass

    def get_collator(self):
        return DataCollatorForCausalLM(
            tokenizer=self.tokenizer,
            max_length=self.model_args.max_sequence_length,
            pad_to_multiple_of=8,
            padding="max_length",
            return_tensors="pt",
        )
    
    def evaluation_step(self, batch):
        pass

    

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()

        epoch_tqdm = tqdm(
            self.eval_dataloader,
            disable=not self.accelerator.is_local_main_process,
            position=1,
            leave=False,
        )
        step_outputs = []
        for step, batch in enumerate(epoch_tqdm):
            outputs = self.evaluation_step(batch)
            step_outputs.append(outputs)

        eval_outputs = self.accelerator.gather_for_metrics(step_outputs)

        if self.accelerator.is_local_main_process:
            eval_outputs = collate_dictlist(eval_outputs)
            eval_results = self.collate_evaluation(eval_outputs)
            eval_results = {f"eval/{k}": v for k, v in eval_results.items()}
            pprint(eval_results)


