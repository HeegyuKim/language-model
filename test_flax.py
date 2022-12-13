from transformers import FlaxAutoModelForCausalLM, AutoTokenizer, pipeline


model_path = "./checkpoint/kogpt2-base-v2"
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = FlaxAutoModelForCausalLM.from_pretrained(model_path)

def generate(prefix: str):
    ids = tokenizer.encode(prefix, return_tensors="jax", padding=False)
    print(ids)
    print(tokenizer.decode(ids[0], skip_special_tokens=False))
    outs = model.generate(input_ids=ids, do_sample=True, top_p=1.0, max_length=128, pad_token_id=2)
    print(outs)
    print(tokenizer.batch_decode(outs.sequences))

# generate("0: 만약 오늘이 ")
# generate("오늘 정부가 발표한 내용에 따르면")
generate("수학이란 학자들의 정의에 따라 ")
# generate("영상 보는데 너무 웃겨 ")