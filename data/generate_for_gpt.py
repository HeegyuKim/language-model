from transformers import AutoTokenizer
from datasets import load_dataset, interleave_datasets
import os
import jsonlines
from tqdm import tqdm


dataset_paths = {
    # "dialog": [
    #     "heegyu/aihub_sns_dialog_gpt",
    #     "heegyu/nikl_online_conv_2022_gpt",
    # ],
    "text": [
        "heegyu/kowikitext",
        # "heegyu/namuwiki-sentences",
        ]
}


class GPTBlockBuilder:
    def load_dataset(self, name):
        dataset = load_dataset(name, split="train")
        name_vars = ["sentence", "dialog", "spoken"]

        for var in name_vars:
            if var in dataset.column_names:
                return dataset.rename_column(var, "text")

        return dataset

    def build_block(self, dataset, key: str):
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

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.bos_token_id = bos_token_id
        self.block_size = block_size

        os.makedirs(output_dir, exist_ok=True)

        for k, paths in dataset_paths.items():
            for path in paths:
                name = path.replace("/", "__")
                with jsonlines.open(f"{output_dir}/{name}.jsonl", "w") as f:
                    dataset = self.load_dataset(path)
                    for block in tqdm(self.build_block(dataset, k), desc=path):
                        f.write({"ids": block})
                    

if __name__ == "__main__":
    builder = GPTBlockBuilder()
    builder.main(
        tokenizer="skt/kogpt2-base-v2",
        bos_token_id=0,
        block_size=1024
    )