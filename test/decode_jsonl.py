import jsonlines
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("heegyu/kogpt-j-base")

with jsonlines.open("/data/dialog-v1-vocab51k-block1024/train/heegyu__aihub_daily_conv_2022_gpt.jsonl") as f:
    item = next(iter(f))

special_tokens_dict = {'additional_special_tokens': [f'<unused{i}>' for i in range(1, 10)]}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

print(tokenizer.tokenize("<unused1>어 그래요?</s>"))
print(tokenizer.decode(item["input_ids"], skip_special_tokens=False))