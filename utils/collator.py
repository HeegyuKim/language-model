
import random
import warnings
from collections.abc import Mapping
from collections import defaultdict
from dataclasses import dataclass, field
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


@dataclass
class LanguageModelCollator(object):
    tokenizer: PreTrainedTokenizerBase
    padding_side: str = "right"
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    is_encoder_decoder: bool = False
    padding_feature_keys: List[str] = field(
        default_factory=["input_ids", "decoder_input_ids", "labels", "attention_mask", "decoder_attention_mask"]
    )

    def __collate(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        out = defaultdict(list)
        for item in out:
            for k, v in item.items():
                out[k].append(v)
        return out

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        features = self.__collate(features)
        padding_features = {}

        for k in self.padding_feature_keys:
            padding_features[k] = features[k]
        
        self.tokenizer.padding_side = padding_side
        batch = self.tokenizer.pad(
            padding_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        for k in features.keys():
            if k not in self.padding_feature_keys:
                batch[k] = features[k]
        
        return batch


@dataclass
class DataCollatorForCausalLM:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.
    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        model ([`PreTrainedModel`]):
            The model that is being trained. If set and has the *prepare_decoder_input_ids_from_labels*, use it to
            prepare the *decoder_input_ids*
            This is useful when using *label_smoothing* to avoid calculating loss twice.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single sequence
              is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
              lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (`int`, *optional*, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    padding_feature_keys: List[str] = field(
        default_factory=lambda: ["input_ids", "decoder_input_ids", "labels", "attention_mask", "decoder_attention_mask"]
    )

    def __collate(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        out = defaultdict(list)
        for item in features:
            for k, v in item.items():
                out[k].append(v)
        return out

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors

        labels = [x['labels'] for x in features] if "labels" in features[0] else None

        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            if self.padding == True or self.padding == "longest":
                max_label_length = max(len(l) for l in labels)
            elif self.padding == "max_length":
                max_label_length = self.max_length

            if self.pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)

        features = self.__collate(features)
        padding_features = {}

        for k in self.padding_feature_keys:
            if k in features:
                padding_features[k] = features[k]

        # print(padding_features)
        # print(features)

        padding_features = self.tokenizer.pad(
            padding_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        features.update(padding_features)

        return dict(features)