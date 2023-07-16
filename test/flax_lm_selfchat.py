from transformers import FlaxAutoModelForCausalLM, AutoTokenizer
from pprint import pprint
import jax

print('start model loading')
model_name = '/data/checkpoint/dialog/heegyu/ajoublue-gpt2-base/checkpoint-epoch-182759-last/'
model_name = '/data/checkpoint/dialog/heegyu/ajoublue-gpt2-medium/checkpoint-epoch-182759-last/'

with jax.default_device(jax.devices("cpu")[0]):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = FlaxAutoModelForCausalLM.from_pretrained(model_name)

print('start test generation')
generation_args = dict(
    # repetition_penalty=1.3,
    # no_repeat_ngram_size=4,
    pad_token_id=0,
    eos_token_id=7,
    max_length=128,
    min_length=64,
    do_sample=True,
    early_stopping=False,
    top_p=0.9
)

tokenizer.padding_side = "left"
tokenizer.truncation_side = "left"

def generate(prompt, **kwargs):
    inputs = tokenizer(prompt, padding=True, truncation=True, return_tensors="np", max_length=512)

    if "min_new_tokens" in kwargs:
        min_new_tokens = kwargs["min_new_tokens"]
        del kwargs["min_new_tokens"]
        kwargs["min_length"] = inputs["input_ids"].shape[-1] + min_new_tokens

    outs = model.generate(**inputs, **kwargs)
    outs = tokenizer.batch_decode(outs.sequences, skip_special_tokens=False)
    pprint(outs)


generate("<s><unused1>", **generation_args)
generate("<s><unused1>", **generation_args)
generate("<s><unused1>", **generation_args)