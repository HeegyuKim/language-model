from torch.utils.data import Dataset
import jsonlines
from dataclasses import dataclass
import os
import random


@dataclass
class InfiniteFileIterator:
    file: str
    tokenizer: Any

    def __iter__(self):
        
        while True:
            with jsonlines.open(self.file) as f:
                for item in f:
                    yield item


class PLMDataset(Dataset):

    def __init__(self, dir_path: str) -> None:
        super().__init__()
        self.files = [os.path.join(dir_path, p) for p in os.listdir(dir_path)]
        
        weights = [os.path.getsize(f) for f in self.files]
        wsum = sum(weights)
        
        # 파일 크기로 가중치를 줍니다
        self.weights = [w / wsum for w in weights]

    def __iter__(self):
        iters = [iter(InfiniteFileIterator(f)) for f in self.files]

        while True:
            item_iter = random.choices(iters, self.weights, k=1)[0]
            item = next(item_iter)
            yield item


if __name__ == "__main__":
    dataset = PLMDataset("data/test/")
    print(dataset.files, dataset.weights)
    dataset_iter = iter(dataset)
    
    for i, item in enumerate(dataset_iter):
        print(item["text"][:10])
        if i == 100:
            break