from transformers import AutoTokenizer
from datasets import load_dataset, interleave_datasets, disable_caching
import os
import jsonlines
from tqdm import tqdm
from typing import List
from dataclasses import dataclass


disable_caching()

def remove_speaker(x):
    text = x["text"]
    for i in range(4):
        text = text.replace(f"{i} : ", "")

    return {
        "text": text
    }

@dataclass
class GPTBlockBuilder:
    dataset_paths: List[str]
    tokenizer: str
    block_size: int
    output_dir: str
    cache_dir: str
    split: str

    def load_dataset(self, name):
        ds = load_dataset(name, split=self.split, cache_dir=self.cache_dir)
        name_vars = ["sentence", "dialog", "spoken", "document", "form", "content"]

        for var in name_vars:
            if var in ds.column_names:
                ds = ds.rename_column(var, "text")
                break

        return ds

    def build_block(self, dataset, key: str = "text"):
        ids = []
        for item in dataset:
            next_ids = self.tokenizer.encode(item[key])
            ids.append(self.bos_token_id)
            ids.extend(next_ids)

            while len(ids) >= self.block_size:
                yield ids[:self.block_size]
                ids = ids[self.block_size:]


    def build(self):
        tokenizer_name = self.tokenizer.replace("/", "__")
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer)
        self.bos_token_id = self.tokenizer.bos_token_id

        os.makedirs(f"{self.output_dir}/", exist_ok=True)

        for path in self.dataset_paths:
            name = path.replace("/", "__")
            with jsonlines.open(f"{self.output_dir}/{name}.jsonl", "w") as f:
                dataset = self.load_dataset(path)
                for block in tqdm(self.build_block(dataset), desc=path):
                    f.write({"input_ids": block})
                    

if __name__ == "__main__":
    dataset_paths = {
        'train': [
            "heegyu/nikl_daily_dialog_2021_gpt",
            "heegyu/nikl_online_conv_2022_gpt",
            "heegyu/aihub_daily_conv_2022_gpt",
            "heegyu/aihub_twitter_dialog_gpt",
            "heegyu/aihub_emotional_dialog_gpt",
        ],
        'test': [
            "heegyu/aihub_daily_conv_2022_gpt",
        ]
    }

    for split, paths in dataset_paths.items():
        GPTBlockBuilder(
            dataset_paths=paths,
            split=split,
            tokenizer="heegyu/kogpt-j-base",
            block_size=1024,
            output_dir=f"/data/dialog-v1-vocab51k-block1024/{split}",
            cache_dir="/data/.cache"
        ).build()