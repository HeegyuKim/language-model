from torch.utils.data import Dataset, IterableDataset
import jsonlines
from dataclasses import dataclass
import os
import random
from transformers import PreTrainedTokenizer
from traceback import print_exc



@dataclass
class InfiniteFileIterator:
    file: str
    tokenizer: PreTrainedTokenizer
    block_size: int

    def __iter__(self):
        
        while True:
            with jsonlines.open(self.file) as f:
                block = []
                for item in f:
                    text = item["text"]
                    if len(text) <= 32:
                        continue
                    
                    try:
                        ids = self.tokenizer.encode(text)
                        block.append(self.tokenizer.bos_token_id)
                        block.extend(ids)

                        while len(block) >= self.block_size:
                            yield {
                                "input_ids": block[:self.block_size]
                            }
                            block = block[self.block_size:]

                    except:
                        # UnicodeDecode error
                        print_exc()
                        print("file", self.file)


class PLMDataset(IterableDataset):

    def __init__(self, dir_path: str, tokenizer: PreTrainedTokenizer, block_size: int) -> None:
        super().__init__()
        self.files = [os.path.join(dir_path, p) for p in os.listdir(dir_path)]
        self.iters = [iter(InfiniteFileIterator(f, tokenizer, block_size)) for f in self.files]
        
        weights = [os.path.getsize(f) for f in self.files]
        wsum = sum(weights)
        
        # 파일 크기로 가중치를 줍니다
        self.weights = [w / wsum for w in weights]

    def __iter__(self):
        while True:
            item_iter = random.choices(self.iters, self.weights, k=1)[0]
            item = next(item_iter)
            item["labels"] = item["input_ids"]
            yield item


if __name__ == "__main__":
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("heegyu/kogpt-neox-small")
    dataset = PLMDataset("data/test/", tokenizer, 32)

    print(dataset.files, dataset.weights)
    dataset_iter = iter(dataset)
    
    for i, item in enumerate(dataset_iter):
        print(tokenizer.decode(item["input_ids"]), len(item["input_ids"]), len(item["labels"]))
        if i == 100:
            break