from transformers import AutoTokenizer
from datasets import load_dataset, interleave_datasets
import os
import jsonlines
from tqdm import tqdm


datasets = [
    # For tiny
    # "heegyu/kowikitext",
    # "heegyu/namuwiki-extracted",
    # "heegyu/aihub_sns_dialog_gpt",
    "heegyu/nikl_messenger_dialog_gpt",
    # "heegyu/aihub_spoken_2021",

    # For small
    # "heegyu/nikl_spoken",
    # "heegyu/nikl_written",
    # "heegyu/nia_web",
    # "heegyu/korean-petitions",
    "heegyu/nikl_daily_dialog_v1.2",

    # For base
    "heegyu/nikl_news",
    
]

def remove_speaker(x):
    text = x["text"]
    for i in range(4):
        text = text.replace(f"{i} : ", "")

    return {
        "text": text
    }


class GPTBlockBuilder:
    def load_dataset(self, name):
        ds = load_dataset(name, split="train", use_auth_token=os.environ["HF_TOKEN"])
        name_vars = ["sentence", "dialog", "spoken", "content", "form"]

        for var in name_vars:
            if var in ds.column_names:
                ds = ds.rename_column(var, "text")
                if var == "dialog":
                    ds = ds.map(remove_speaker)
                break

        return ds

    def build_block(self, dataset, key: str = 'text'):
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
             bos_token_id: int, 
             block_size: int,
             output_dir: str = "gpt_block"
             ):
        tokenizer_name = tokenizer.replace("/", "__")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.bos_token_id = bos_token_id
        self.block_size = block_size

        os.makedirs(f"{output_dir}/{tokenizer_name}", exist_ok=True)

        for path in datasets:
            name = path.replace("/", "__")
            with jsonlines.open(f"{output_dir}/{tokenizer_name}/{name}.jsonl", "w") as f:
                dataset = self.load_dataset(path)
                for block in tqdm(self.build_block(dataset), desc=path):
                    f.write({"input_ids": block})
                    

if __name__ == "__main__":
    builder = GPTBlockBuilder()
    builder.main(
        tokenizer="heegyu/kogpt-j-base",
        bos_token_id=1,
        block_size=1024
    )