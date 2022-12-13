from transformers import AutoTokenizer
from datasets import load_dataset, interleave_datasets, disable_caching
import os
import jsonlines
from tqdm import tqdm


disable_caching()


dataset_paths = [
    # For tiny
    "heegyu/kowikitext",
    "heegyu/namuwiki-extracted",
    "heegyu/aihub_sns_dialog_gpt",
    "heegyu/nikl_messenger_dialog_gpt",
    "heegyu/aihub_spoken_2021",

    # For small
    "heegyu/nikl_spoken",
    "heegyu/nikl_written",
    "heegyu/nia_web",
    "heegyu/korean-petitions",
    "heegyu/nikl_daily_dialog_v1.2",

    # For base
    "heegyu/nikl_news",
    "heegyu/nia_book",
    "Bingsu/KcBERT_Pre-Training_Corpus",
]


class GPTBlockBuilder:
    def load_dataset(self, name):
        dataset = load_dataset(name, split="train")
        name_vars = ["sentence", "dialog", "spoken", "document"]

        for var in name_vars:
            if var in dataset.column_names:
                return dataset.rename_column(var, "text")

        return dataset

    def build_block(self, dataset, key: str = "text"):
        ids = []
        for item in dataset:
            next_ids = self.tokenizer.encode(item[key])
            ids.append(self.bos_token_id)
            ids.extend(next_ids)

            while len(ids) >= self.block_size:
                yield ids[:self.block_size]
                ids = ids[self.block_size:]


    def main(self, 
             tokenizer: str, 
             block_size: int,
             output_dir: str = "gpt_block"
             ):

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.bos_token_id = self.tokenizer.bos_token_id
        self.block_size = block_size

        os.makedirs(output_dir, exist_ok=True)

        for path in dataset_paths:
            name = path.replace("/", "__")
            with jsonlines.open(f"{output_dir}/{name}.jsonl", "w") as f:
                dataset = self.load_dataset(path)
                for block in tqdm(self.build_block(dataset), desc=path):
                    f.write({"input_ids": block})
                    

if __name__ == "__main__":
    builder = GPTBlockBuilder()
    # builder.main(
    #     tokenizer="heegyu/kogpt-neox-small",
    #     block_size=1024
    # )
    builder.main(
        tokenizer="heegyu/kogpt-j-base",
        block_size=1024
    )