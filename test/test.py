from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# generator = pipeline('text-generation', model='./huggingface/kogpt-neox-tiny', device='cuda:0')
generator = pipeline('text-generation', model='../checkpoint/ctrl/ctrl-gpt2-imdb/epoch-4-last')

def generate(prefix: str):
    print(generator(prefix, do_sample=True, top_p=1.0, repetition_penalty=1.2, max_length=32)[0]["generated_text"])

generate("positive this is my ")
generate("negative this is my ")
# generate("오늘 정부가 발표한 내용에 따르면")
# generate("수학이란 학자들의 정의에 따라")
# generate("영상 보는데 너무 웃겨 ")