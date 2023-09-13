import os
os.environ["XRT_TPU_CONFIG"]="localservice;0;localhost:51011"

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

from pprint import pprint

from transformers import HfArgumentParser
from task import nsmc, director, ctrl, klue, dexpert, sequence_classification,  gpt, \
    detox, koalpaca, reward

TASKS = {
    "nsmc": nsmc.NSMCTask,
    "director": director.DirectorTask,
    "ctrl": ctrl.CTRLTask,
    "klue-ynat": klue.YNATTask,
    "klue-sts": klue.STSBinaryTask,
    # "nia-summ": summarization.NiaSummarizationTask,
    # "nia-dialog": dialog.NiaDialogTask,
    # "nia-dialog-v2": dialog.NiaDialogTaskV2,
    "dexpert-toxic": dexpert.ToxicDExpertTask,
    "dexpert-non-toxic": dexpert.NonToxicDExpertTask,
    "news-category-top10": sequence_classification.NewsCategoryClassificationTask,
    "toxic-token-classification": detox.ToxicSpanDetectionTask,
    "toxic-sequence-classification": detox.ToxicSequenceClassificationTask,
    "gpt": gpt.GPTTask,
    "gpt-finetuning": gpt.CausalFineTuningTask,
    "gpt-lyrics": gpt.LyricsGPTTask,
    "koalpaca": koalpaca.KoAlpacaTask,
    "gorani": gpt.GoraniTask,
    "reward": reward.RewardTask
    }


def main():
    parser = HfArgumentParser(
        (TrainingArguments, DataTrainingArguments, ModelArguments)
    )
    training_args, data_args, model_args = parser.parse_args_into_dataclasses()
    args = parser.parse_args()

    set_seed(args.seed)

    os.environ["WANDB_NAME"] = args.run_name
    accelerator = Accelerator(log_with="wandb")
    accelerator.init_trackers(
        args.project,
        config=args
    )
    task = TASKS[args.task](accelerator, training_args, data_args, model_args)
    task.setup()
    if args.do_train:
        task.train()
    elif args.do_eval:
        task.evaluate(0, 0)

    accelerator.end_training()

if __name__ == "__main__":
    main()