from transformers import FlaxAutoModelForCausalLM, AutoTokenizer, pipeline

import jax
import jax.numpy as jnp

# Global flag to set a specific platform, must be used at startup.
jax.config.update('jax_platform_name', 'cpu')


# model_path = "./checkpoint/gpt-j-base-v0-lr6e-4-batch8/checkpoint-60000"
model_path = "./checkpoint/gpt-j-base-v1-24L-lr6e-4-batch4-rev1/checkpoint-625000"
tokenizer = AutoTokenizer.from_pretrained(model_path)
print("hi")
# model = FlaxAutoModelForCausalLM.from_pretrained(model_path, _do_init=False)[0]
model = FlaxAutoModelForCausalLM.from_pretrained(model_path)
# print(model)

def generate(prefix: str, do_sample=True, max_new_tokens=64):
    ids = tokenizer.encode(prefix, return_tensors="jax", padding=False)
    print(len(ids[0]))
    # print(tokenizer.decode(ids[0], skip_special_tokens=False))
    outs = model.generate(input_ids=ids, do_sample=do_sample, top_p=1.0, temperature=1.5, max_new_tokens=max_new_tokens)
    # print(outs)?
    print(tokenizer.batch_decode(outs.sequences))

# generate("0: 만약 오늘이 ")
# generate("오늘 정부가 발표한 내용에 따르면")
# generate("수학이란 학자들의 정의에 따라 ")
# generate("영상 보는데 너무 웃겨 ")

questions = [
    """제시된 단어를 보고 대통령을 욕하는 창의적인 문장을 만들어보세요
사퇴=대통령은 하는 것도 없으면서 ㅉㅉ 사퇴해라
검찰=대통령 임기 언제끝나냐 각종비리 검찰수사 들어가야한다.
탄핵=""",
    """손흥민에 대한 뉴스 제목을 작성하세요
뉴스 제목: 손흥민, EPL 득점왕 달성할 수 있을까? 전문가들의 예상은?
뉴스 제목: 손흥민 풀타임, 이강인 강인함…‘가나전’ 승리 가나요
뉴스 제목: 후배들을 기 살려준 손흥민 "쫄지마, 너희도 잘하는 선수야"
뉴스 제목: """,
"""이제부터 제가 재미있는 소설을 작성해보겠습니다.
내용: 요즘 우리 집안 분위기가 심상치 않다. """,
"""노래 제목과 장르를 보고, 사람들이 좋아할 만한 노래 가사를 써주세요.

노래 제목: 그대라는 시
장르: 발라드
가사:
언제부터인지 그대를 보면
운명이라고 느꼈던 걸까
밤하늘의 별이 빛난 것처럼
오랫동안 내 곁에 있어요
그대라는 시가 난 떠오를 때마다
외워두고 싶어 그댈 기억할 수 있게
슬픈 밤이 오면 내가 그대를 지켜줄게
내 마음 들려오나요 잊지 말아요

노래 제목: 밤편지
장르: 발라드/어쿠스틱
가사: 
이 밤 그날의 반딧불을 당신의
"""
]

for q in questions:
    generate(q.strip())