from transformers import pipeline

print('start model loading')
# model_name = '../checkpoint/koalpaca/ajoublue-gpt2-medium/epoch-4-last/'
model_name = '../checkpoint/gorani/heegyu__ajoublue-gpt2-medium/epoch-2/'
generator = pipeline(
    'text-generation',
    model=model_name,
    tokenizer=model_name
)

def query(instruction, input=None):
    # if input:
    #     prompt = f"<usr>{instruction}<sys>{input}\n<bot>"
    # else:
    #     prompt = f"<usr>{instruction}\n<bot>"
    prompt = instruction
    
    print(generator(
        prompt,
        do_sample=True,
        top_p=0.9,
        early_stopping=True,
        max_length=256,
    )[0]['generated_text'])


# query("슈카월드에 대해서 알아?")
# query("세상에서 가장 유명한 사람은?")
# query("알버트 아인슈타인에 대해서 설명해줘")
# query("다음 영화에 대해서 설명해줘", "기생충")
# query("섭씨 온도를 화씨로 변경해줘", "섭씨 온도: 15도")


def query_dialog(prompt):
    prompt = f"{prompt}".strip()
    
    print(generator(
        prompt,
        do_sample=True,
        top_p=1.0,
        early_stopping=True,
        max_length=256,
    )[0]['generated_text'])

query_dialog("""
<usr> 알버트 아인슈타인에 대해서 설명해줘
""".strip())

# query_dialog("""
# <usr> 우리가 나눴던 대화에 내용이 이어지도록 대답해.
# <sys> 알버트 아인슈타인에 대해서 설명해줘. 알버트 아인슈타인은 상대성 이론을 바탕으로 하는 상대성이론 연구로 유명한 독일의 이론 물리학자였습니다. 그는 빛의 속도가 입자보다 크거나 같다는 상대성 이론을 사용하여 빛의 속도가 입자보다 크거나 같다는 것을 증명했습니다. 그 사람은 지금 살아있어? 
# """.strip())

# query_dialog("""
# <usr> 1: 알버트 아인슈타인에 대해서 설명해줘
# <bot> 1: 알버트 아인슈타인은 상대성 이론을 바탕으로 하는 상대성이론 연구로 유명한 독일의 이론 물리학자였습니다. 그는 빛의 속도가 입자보다 크거나 같다는 상대성 이론을 사용하여 빛의 속도가 입자보다 크거나 같다는 것을 증명했습니다
# <usr> 2: 그 사람은 지금 살아있어?
# <bot> 2: 아니오 그는 이미 사망했습니다.
# <usr> 3: 그렇구나 안타깝네 나도 그처럼 되려면 뭘 배워야 할까?
# <bot> 3: 
# """.strip())

query_dialog("""
<usr> 알버트 아인슈타인에 대해서 설명해줘
<bot> 알버트 아인슈타인은 상대성 이론을 바탕으로 하는 상대성이론 연구로 유명한 독일의 이론 물리학자였습니다.
<usr> 그 사람은 지금 살아있어?
<bot> 아니오 그는 이미 사망했습니다.
<usr> 그와 같은 훌륭한 물리학자가 되려면 어떻게 해야할까?
<bot> 
""".strip())


# query_dialog("""
# <usr> 아래 대화와 대답을 읽고, 대화3에서 요구하는 내용을 만족하는 대답3을 작성해.
# <sys> 대화1: 알버트 아인슈타인에 대해서 설명해줘. 
# 대답1: 알버트 아인슈타인은 상대성 이론을 바탕으로 하는 상대성이론 연구로 유명한 독일의 이론 물리학자였습니다. 그는 빛의 속도가 입자보다 크거나 같다는 상대성 이론을 사용하여 빛의 속도가 입자보다 크거나 같다는 것을 증명했습니다. 
# 대화2: 그 사람은 지금 살아있어?. 
# 대답2: 아니오 그는 이미 사망했습니다. 
# <bot> 대화3: 그렇구나 안타깝네 나도 그처럼 되려면 뭘 배워야 할까?
# 대답3:
# """.strip())

query_dialog("""
<usr> 세상에서 가장 유명한 사람은?
<bot> 세상에서 가장 유명한 사람은 주관적인 질문이라 대답을 하기 힘듭니다.
<usr> 그러면 한국에서는 누가 유명할까?
<bot> 
""")
             
# query_dialog("""
# <usr> 세상에서 가장 유명한 사람은?
# <bot> 세상에서 가장 유명한 사람은 주관적인 질문이라 대답을 하기 힘듭니다.
# <usr> 그러면 한국에서는 누가 유명할까?
# <bot> 이승만씨는 우리나라에서 가장 유명한 인물 중 한 분입니다
# <usr> 그 사람은 어떤 사람이야?
# <bot> 이승만 씨는 한국의 초대 대통령이었습니다
# """)
             
query_dialog("""
<usr> 세상에서 가장 유명한 사람은?
<bot> 세상에서 가장 유명한 사람은 주관적인 질문이라 대답을 하기 힘듭니다.
<usr> 그러면 한국에서는 누가 유명할까?
<bot> 이승만씨는 우리나라에서 가장 유명한 인물 중 한 분입니다
<usr> 그 사람은 어떤 사람이야?
<bot> 이승만 씨는 한국의 초대 대통령이었습니다
<usr> 좀 더 자세히 이야기해줘.
<bot>
""")
             
query_dialog("""
<usr> 안녕하세요. 넌 한국어 챗봇 고라니야. 너는 내가 묻는 질문에 답하고 지시사항에 맞는 대답을 해야해.
<bot> 안녕하세요, 저는 고라니입니다. 지시하신 내용을 따르겠습니다.
<usr> 오늘 날씨는 어때?
<bot> 
""")

query_dialog("""
<usr> 안녕하세요. 넌 한국어 챗봇 고라니야. 너는 내가 묻는 질문에 답하고 지시사항에 맞는 대답을 해야해.
<bot> 안녕하세요, 저는 고라니입니다. 지시하신 내용을 따르겠습니다.
<usr> 오늘 날씨는 어때?
<bot> 오늘 날씨는 화창합니다.
<usr> 이런 날 뭐하고 놀면 좋을까?
<bot> 
""")

query_dialog("""
<usr> 안녕하세요. 넌 한국어 챗봇 고라니야. 너는 내가 묻는 질문에 답하고 지시사항에 맞는 대답을 해야해.
<bot> 안녕하세요, 저는 고라니입니다. 지시하신 내용을 따르겠습니다.
<usr> 너는 누구야?
<bot> 
""")
query_dialog("""
<usr> 넌 한국어 챗봇 고라니야. 너는 내가 묻는 질문에 답하고 지시사항에 맞는 대답을 해야해.
<bot> 안녕하세요, 저는 고라니입니다. 지시하신 내용을 따르겠습니다.
<usr> 김희규에 대해서 알려줘
<bot> 
""")
             
# query_dialog("""
# <usr> 대화를 참고하여 나의 마지막 질문에 '너'가 해야할 말을 작성해.
# <sys> 나: 알버트 아인슈타인에 대해서 설명해줘. 
# 너: 알버트 아인슈타인은 상대성 이론을 바탕으로 하는 상대성이론 연구로 유명한 독일의 이론 물리학자였습니다. 그는 빛의 속도가 입자보다 크거나 같다는 상대성 이론을 사용하여 빛의 속도가 입자보다 크거나 같다는 것을 증명했습니다. 
# 나: 그 사람은 지금 살아있어?. 
# 너: 아니오 그는 이미 사망했습니다. 
# 나: 그렇구나 안타깝네 나도 그처럼 되려면 뭘 배워야 할까?
# """.strip())
