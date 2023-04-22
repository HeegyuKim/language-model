from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# generator = pipeline('text-generation', model='../checkpoint/gpt-finetuning/skt__kogpt2-base-v2-heegyu__naver_webnovel/epoch-9')
generator = pipeline('text-generation', model='../checkpoint/gpt-lyrics/heegyu__ajoublue-gpt2-medium/epoch-6')

def generate(prefix: str):
    print(generator(prefix, do_sample=True, top_p=0.9, temperature=1.5, max_length=512, early_stopping=True)[0]["generated_text"])

generate("제목: 밤편지\n아티스트: 아이유\n가사:\n")
generate("제목: Love\n아티스트: 빅뱅\n가사:\n")
# generate("오늘 정부가 발표한 내용에 따르면")
# generate("수학이란 학자들의 정의에 따라")
# generate("영상 보는데 너무 웃겨 ")