from transformers import AutoTokenizer
from datasets import load_dataset, interleave_datasets, disable_caching
import os
import jsonlines
from tqdm import tqdm


disable_caching()


dataset_paths = [
    # For tiny
    "heegyu/kowikitext",
    # "heegyu/namuwiki-extracted",
    # "heegyu/aihub_sns_dialog_gpt",
    # "heegyu/nikl_messenger_dialog_gpt",
    # "heegyu/aihub_spoken_2021",

    # For small
    # "heegyu/nikl_spoken",
    # "heegyu/nikl_written",
    # "heegyu/nia_web",
    # "heegyu/korean-petitions",
    # "heegyu/nikl_daily_dialog_v1.2",

    # For base
    # "heegyu/nikl_news",
    # "heegyu/nia_book",
    # "Bingsu/KcBERT_Pre-Training_Corpus",
]

# for dialog
# dataset_paths = [
#     "heegyu/nikl_daily_dialog_2021_gpt",
#     "heegyu/nikl_online_conv_2022_gpt",
#     "heegyu/aihub_daily_conv_2022_gpt",
#     "heegyu/aihub_twitter_dialog_gpt",
#     "heegyu/aihub_emotional_dialog_gpt",
# ]
# split="train"

dataset_paths = [
    "heegyu/aihub_daily_conv_2022_gpt",
]
split="test"


# True 면, 데이터 전처리 할 때 speaker id 안지운다
is_dialog = True


def remove_speaker(x):
    text = x["text"]
    for i in range(4):
        text = text.replace(f"{i} : ", "")

    return {
        "text": text
    }

class GPTBlockBuilder:
    def load_dataset(self, name, cache_dir):
        ds = load_dataset(name, split=split, cache_dir=cache_dir)
        name_vars = ["sentence", "dialog", "spoken", "document", "form", "content"]

        for var in name_vars:
            if var in ds.column_names:
                ds = ds.rename_column(var, "text")
                if not is_dialog and var == "dialog":
                    ds = ds.map(remove_speaker)
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


    def main(self, 
             tokenizer: str, 
             block_size: int,
             output_dir: str,
             cache_dir: str
             ):
        tokenizer_name = tokenizer.replace("/", "__")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.bos_token_id = self.tokenizer.bos_token_id
        self.block_size = block_size

        os.makedirs(f"{output_dir}/", exist_ok=True)

        for path in dataset_paths:
            name = path.replace("/", "__")
            with jsonlines.open(f"{output_dir}/{name}.jsonl", "w") as f:
                dataset = self.load_dataset(path, cache_dir)
                for block in tqdm(self.build_block(dataset), desc=path):
                    f.write({"input_ids": block})
                    

if __name__ == "__main__":
    builder = GPTBlockBuilder()
    # builder.main(
    #     tokenizer="heegyu/kogpt-neox-small",
    #     block_size=1024
    # )
    # builder.main(
    #     tokenizer="heegyu/kogpt-j-base",
    #     block_size=1024,
    #     output_dir="/data2/v1-vocab51k-block1024",
    #     cache_dir="/data2/.cache"
    # )

    builder.main(
        tokenizer="heegyu/kogpt-j-base",
        block_size=1024,
        output_dir="/data2/dialog-v1-vocab51k-block1024/test",
        cache_dir="/data2/.cache"
    )