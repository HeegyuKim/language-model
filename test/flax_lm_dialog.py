from transformers import FlaxAutoModelForCausalLM, AutoTokenizer
from pprint import pprint


print('start model loading')
model_name = '../checkpoint/gpt-finetuning/heegyu__ajoublue-gpt2-medium-heegyu__naver_webnovel/epoch-9'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = FlaxAutoModelForCausalLM.from_pretrained(model_name)

print('start test generation')
generation_args = dict(
    # repetition_penalty=1.3,
    no_repeat_ngram_size=4,
    # eos_token_id=375, # \n
    eos_token_id=2, # </s>
    pad_token_id=0,
    max_new_tokens=16,
    min_new_tokens=8,
    do_sample=False,
    top_p=0.7,
    # num_beams=4,
    early_stopping=True
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
    outs = tokenizer.batch_decode(outs.sequences, skip_special_tokens=True)
    pprint(outs)


# print(generate(
#     ["0 : **는 게임 좋아하니</s>1 :"],
#     **generation_args
# ))
print(generate(
    [
        "그녀가 내게 말했다\n\"안녕?\""
    ]
    **generation_args
))

# chats = []

# while True:
#     msg = input("나: ")
#     chats.append(f"1: {msg}")
#     prompt = "\n".join(chats) + "\n0: "
#     res = generator(prompt)[0]['generated_text']
#     print(res)
#     res = "0: " + res[len(prompt):]
#     res = res.split("\n")[0]
#     chats.append(res)
