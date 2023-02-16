from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch_xla.core.xla_model as xm
from pprint import pprint
import time

dev = xm.xla_device()
print(dev)
print(xm.get_xla_supported_devices())

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'
model = AutoModelForCausalLM.from_pretrained(model_name).to(dev)
model.config.pad_token_id = model.config.eos_token_id

def generate(prompt: str, tokenizer_args: dict, generate_args: dict):
    start = time.time()

    prompt = tokenizer(prompt, **tokenizer_args, return_tensors="pt")
    prompt = {k: v.to(dev) for k, v in prompt.items()}
    outs = model.generate(**prompt, **generate_args)

    dur = time.time() - start
    print("generation time:", dur)
    pprint(tokenizer.batch_decode(outs, skip_special_tokens=True))


tokenizer_args = {
    "padding": "max_length",
    "max_length": 32
}
generate_args = {
    "do_sample": True,
    "max_new_tokens": 5,
}

generate("Hello, ", tokenizer_args, generate_args)
generate("Hello, My name is...", tokenizer_args, generate_args)
generate("WHAT THE HELL!!!!", tokenizer_args, generate_args)