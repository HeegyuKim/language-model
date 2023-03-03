import torch
from transformers import pipeline

# tokenizer = AutoTokenizer.from_pretrained("gpt2")
# tokenizer.save_pretrained("huggingface/gpt2-toxic")

pipe = pipeline("text-generation", model="./checkpoint/dexpert/dexpert-non-toxic/epoch-2/")

print(pipe("Hi, ", do_sample=True, max_new_tokens=32))

# params = torch.load("checkpoint/dexpert/dexpert-toxic/epoch-1/pytorch_model.bin")
# for k, v in params.items():
#     print(k, v.shape)