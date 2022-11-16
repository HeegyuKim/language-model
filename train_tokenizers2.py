from tokenizers.trainers import BpeTrainer
from tokenizers.models import BPE
from tokenizers import (
    Tokenizer,
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    trainers,
)
from pathlib import Path
from datasets import load_dataset, interleave_datasets
from tokenizers import ByteLevelBPETokenizer


special_tokens = [
    "<pad>",
    "<s>",
    "</s>",
    "<usr>",
    "<sys>",
    "<unk>",
    "<mask>",
    "<|endoftext|>",
] + [f"<unused{i}" for i in range(1, 65)]

tokenizer = Tokenizer(BPE(unk_token="<unk>",))
tokenizer.normalizer = normalizers.NFKC()
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
tokenizer.decoder = decoders.ByteLevel()


def batch_iterator(dataset, batch_size=1000, drop_last: bool = False):
    batch = []

    for item in dataset:
        batch.append(item["text"])

        if len(batch == batch_size):
            yield batch
            batch = []

    if not drop_last and batch:
        yield batch


dataset_paths = ["heegyu/kowikitext", "heegyu/namuwiki-extracted"]
jsonl_files = [
    "nia_sns.jsonl",
    "nia_spoken.jsonl",
    "nia_web.json",
    "nikl_messenger_v2.0.jsonl",
    "nikl_spoken_v1.2.jsonl",
]

datasets = [load_dataset(d, split="train", streaming=True) for d in dataset_paths]
jsonls = [
    load_dataset("json", data_files=f"data/output/{f}", streaming=True)
    for f in jsonl_files
]

dataset = interleave_datasets(datasets + jsonls, stopping_strategy="all_exhausted")

trainer = BpeTrainer(special_tokens=special_tokens)
tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)
tokenizer.save("data/tokenizer-.json")
