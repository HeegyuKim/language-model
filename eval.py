"""

python eval.py \ 
    --model heegyu/kogpt-j-base-summarization \
    --task nia-summ

"""
from transformers import HfArgumentParser, set_seed
from utils.arguments import TrainingArguments, DataTrainingArguments, ModelArguments
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from typing import List, Dict, Any
from datasets import disable_caching, load_dataset
from tqdm.auto import tqdm

import os
import evaluate
from pprint import pprint

from transformers import HfArgumentParser
from task.eval import summarization

TASKS = {
    "nia-summ": summarization.NiaSummarizationTask,
    }


def main():
    parser = HfArgumentParser(
        (TrainingArguments, DataTrainingArguments, ModelArguments)
    )
    training_args, data_args, model_args = parser.parse_args_into_dataclasses()
    args = parser.parse_args()

    set_seed(args.seed)

    os.environ["WANDB_NAME"] = args.run_name
    # accelerator = Accelerator(log_with="wandb")
    # accelerator.init_trackers(
    #     args.project,
    #     config=args
    # )
    task = TASKS[args.task](accelerator, training_args, data_args, model_args)
    task.setup()
    task.evaluate(0, 0)

    # accelerator.end_training()
