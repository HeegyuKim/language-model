from typing import List, Dict
from .base import BaseTask

import torch
import torch.nn.functional as F
import evaluate

from pprint import pprint
from datasets import load_dataset
from utils.collator import RewardModelCollator, Seq2SeqRewardModelCollator
from utils.metric import ConfiguredMetric

from datasets import load_dataset
from .model.t5_reward import T5ForRewardModel


def join_conv(convs):
    lines = []
    for conv in convs:
        if conv["from"] == "human":
            lines.append("Human: " + conv["value"])
        else:
            lines.append("Assistant: " + conv["value"])
    return "\n\n".join(lines)


class RewardTask(BaseTask):
    

    def prepare_dataset(self):
        args = self.data_args
                
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        
        self.tokenizer.truncation_side = "left"

        with self.accelerator.local_main_process_first():
            dataset = load_dataset("heegyu/hh-rlhf-vicuna-format")

            if args.max_train_samples:
                dataset["train"] = dataset["train"].select(range(args.max_train_samples))
            if args.max_eval_samples:
                dataset["test"] = dataset["test"].select(range(args.max_eval_samples))

            dataset = dataset.filter(self.filter_item)
            dataset = dataset.map(self.encode, load_from_cache_file=False, num_proc=8)

        return {
            "train": dataset["train"],
            "validation": dataset["test"]
        }

    def filter_item(self, x):
        chosen, rejected = x["chosen"]["value"].strip(), x["rejected"]["value"].strip()
        if chosen and rejected:
            return True
        else:
            return False

    def encode_item(self, prefix, text):
        batch = self.tokenizer(text, truncation=True, max_length=self.model_args.max_sequence_length)
        # return {f"{prefix}{k}": v for k, v in batch.items()}
        return batch
    
    def encode(self, item):
        chosen = join_conv(item["context"] + [item["instruction"], item["chosen"]])
        rejected = join_conv(item["context"] + [item["instruction"], item["rejected"]])

        chosen = self.encode_item("chosen_", chosen)
        rejected = self.encode_item("rejected_", rejected)
        # return dict(**chosen, **rejected)
        return dict(
            chosen=chosen,
            rejected=rejected
        )

    def get_collator(self):
        return RewardModelCollator(
            tokenizer=self.tokenizer,
            max_length=self.model_args.max_sequence_length,
            pad_to_multiple_of=8,
            padding="max_length",
            return_tensors="pt",
        )

    def step(self, batch):
        # chosen = {k.replace("chosen_", ""): v for k, v in batch.items() if k.startswith("chosen_")}
        # rejected = {k.replace("rejected_", ""): v for k, v in batch.items() if k.startswith("rejected_")}

        # for k, v in batch["chosen"].items():
        #     print("chosen", k, v.shape)

        # for k, v in batch["rejected"].items():
        #     print("rejected", k, v.shape)

        chosen = self.model(**batch["chosen"]).logits.squeeze(1)
        rejected = self.model(**batch["rejected"]).logits.squeeze(1)

        loss = -F.logsigmoid(chosen - rejected).mean()
        # print(loss)
        accuracy = chosen > rejected
        accuracy = accuracy.detach().float().mean()

        return dict(
            loss=loss,
            accuracy=accuracy
        )

    def training_step(self, batch):
        return self.step(batch)

    def evaluation_step(self, batch):
        return self.step(batch)

    def collate_evaluation(self, results: List[Dict]):
        loss = torch.stack(results['loss']).mean().item()
        acc = torch.stack(results['accuracy']).mean().item()
        eval_results = {
            "loss": loss,
            "accuracy": acc
        }
        pprint("evaluation result")
        pprint(eval_results)
        return eval_results


class Seq2SeqRankRewardTask(RewardTask):
    
    def get_model(self, args):
        
        model = T5ForRewardModel.from_pretrained(
            args.model_name_or_path,
            revision=args.revision,
            from_flax=args.from_flax,
            num_labels=1,
            )
        return model

    def get_collator(self):
        return RewardModelCollator(
            tokenizer=self.tokenizer,
            max_length=self.model_args.max_sequence_length,
            pad_to_multiple_of=8,
            padding="max_length",
            return_tensors="pt",
        )

    def encode_item(self, prefix: str, text, decoder):
        max_len = self.model_args.decoder_max_sequence_length if decoder else self.model_args.max_sequence_length
        
        self.tokenizer.truncation_side = 'right' if decoder else 'left'

        batch = self.tokenizer(text, truncation=True, max_length=max_len, add_special_tokens=decoder)
        return {f"{prefix}{k}": v for k, v in batch.items()}
    
    def encode(self, item):
        instruction = join_conv(item["context"] + [item["instruction"]]) + "\n\nAssistant: "
        chosen = item["chosen"]["value"]
        rejected = item["rejected"]["value"]

        instruction = self.encode_item("", instruction, False)
        chosen = self.encode_item("decoder_", chosen, True)
        rejected = self.encode_item("decoder_", rejected, True)

        # return dict(**chosen, **rejected)
        return dict(
            chosen={
                **instruction,
                **chosen,
            },
            rejected={
                **instruction,
                **rejected
            }
        )

    def get_collator(self):
        return Seq2SeqRewardModelCollator(
            tokenizer=self.tokenizer,
            max_length=self.model_args.max_sequence_length,
            decoder_max_length=self.model_args.decoder_max_sequence_length,
            pad_to_multiple_of=8,
            padding="max_length",
            return_tensors="pt",
        )

    def step(self, batch):
        # for k, v in batch["chosen"].items():
        #     print("chosen", k, v.shape)

        # for k, v in batch["rejected"].items():
        #     print("rejected", k, v.shape)

        chosen = self.model(**batch["chosen"]).logits.squeeze(1)
        rejected = self.model(**batch["rejected"]).logits.squeeze(1)

        loss = -F.logsigmoid(chosen - rejected).mean()
        # print(loss)
        accuracy = chosen > rejected
        accuracy = accuracy.detach().float().mean()

        return dict(
            loss=loss,
            accuracy=accuracy
        )