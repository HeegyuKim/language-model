
import random
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from random import randint
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

import numpy as np
import torch

from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy

@dataclass
class SequenceClassificationCollator(object):

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def _pad(self, x: List):
        return self.tokenizer.pad(
            x,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        labels = [x['label'] for x in features]
        if self.return_tensors == 'pt':
            labels = torch.tensor(labels, dtype=torch.long)

        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        
        del batch['label']
        batch['labels'] = labels

        return batch
