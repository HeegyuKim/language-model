from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# generator = pipeline('text-generation', model='../checkpoint/gpt-finetuning/skt__kogpt2-base-v2-heegyu__naver_webnovel/epoch-9')
generator = pipeline('text-generation', model='../checkpoint/gpt-finetuning/heegyu__ajoublue-gpt2-medium-heegyu__naver_webnovel/epoch-9')

def generate(prefix: str):
    print(generator(prefix, do_sample=True, top_p=0.9, temperature=1.5, max_length=1024, repetition_penalty=1.1)[0]["generated_text"])

generate("그녀가 내게 말했다\n\"안녕?\"\n")
generate("오늘 ")
# generate("오늘 정부가 발표한 내용에 따르면")
# generate("수학이란 학자들의 정의에 따라")
# generate("영상 보는데 너무 웃겨 ")