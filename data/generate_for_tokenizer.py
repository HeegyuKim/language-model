from pathlib import Path
import jsonlines
from datasets import load_dataset
from tqdm import tqdm

data_dir = "tokenizer_data"


def write_iter(filename, iter):
    with open(data_dir + "/" + filename, "w", encoding="utf-8") as f:
        for item in tqdm(iter, desc=filename):
            f.write(item["text"])
            f.write("\n")


dataset_paths = [
    # "heegyu/kowikitext",
    # "heegyu/namuwiki-extracted"
]

for d in dataset_paths:
    print(d)
    name = d.split("/")[1] + ".txt"
    dataset = load_dataset(d, split="train")
    write_iter(name, dataset)


jsonl_files = [
    "nia_sns.jsonl",
    "nia_spoken.jsonl",
    # "nia_web.jsonl", # 넘 많다.
    "nikl_messenger_v2.0.jsonl",
    "nikl_spoken_v1.2.jsonl",
]
for file in jsonl_files:
    print(file)
    with jsonlines.open(f"output/{file}") as f:
        write_iter(file + ".txt", f)
