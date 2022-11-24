from tokenizers import ByteLevelBPETokenizer
from datasets import load_dataset, interleave_datasets, concatenate_datasets
import os

special_tokens = [
    "<pad>",
    "<s>",
    "</s>",
    "<usr>",
    "<sys>",
    "<unk>",
    "<mask>",
    "<|endoftext|>",
] + [f"<unused{i}>" for i in range(1, 65)]


dir = "data/tokenizer_data/"
split = "train[:10%]"
# files = [dir + x for x in os.listdir(dir)]
vocab_size = 51200
min_freq = 5


def my_load_dataset(name):
    dataset = load_dataset(name, split=split, use_auth_token=True)
    name_vars = ["sentence", "dialog", "spoken"]

    for var in name_vars:
        if var in dataset.column_names:
            return dataset.rename_column(var, "text")

    return dataset


dataset_paths = [
    # dialog
    "heegyu/aihub_daily_conv_2022_gpt",
    "heegyu/aihub_sns_dialog_gpt",
    "heegyu/nikl_online_conv_2022_gpt",

    # written
    "heegyu/aihub_web_2021", # sentence
    "heegyu/namuwiki-sentences", # text
    "heegyu/kowikitext", # text

    # spoken
    "heegyu/aihub_spoken_2021", # sentence
]
if __name__ == "__main__":
    print("train bpe tokenizers from", dataset_paths)

    dataset_paths = map(my_load_dataset, dataset_paths)
    dataset = interleave_datasets(list(dataset_paths), stopping_strategy="all_exhausted")
    # dataset = concatenate_datasets(list(dataset_paths))

    def batch_iterator(dataset, batch_size=1000, drop_last: bool = False):
        batch = []

        for item in dataset:
            batch.append(item["text"])

            if len(batch) == batch_size:
                yield batch
                batch = []

        if not drop_last and batch:
            yield batch

    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train_from_iterator(
        batch_iterator(dataset),
        vocab_size=vocab_size,
        min_frequency=min_freq,
        special_tokens=special_tokens,
    )

    dir_name = f"models/tokenizer-{vocab_size // 1000}k/"
    os.makedirs(dir_name, exist_ok=True)
    tokenizer.save_model(dir_name)
