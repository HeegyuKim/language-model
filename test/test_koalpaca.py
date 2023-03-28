from transformers import pipeline

print('start model loading')
model_name = '../checkpoint/koalpaca/ajoublue-gpt2-medium/epoch-4-last/'
generator = pipeline(
    'text-generation',
    model=model_name,
    tokenizer=model_name
)

def query(instruction, input=None):
    if input:
        prompt = f"<usr>{instruction}\n{input}\n<sys>"
    else:
        prompt = f"<usr>{instruction}\n<sys>"
    
    print(generator(
        prompt,
        do_sample=True,
        top_p=0.9,
        early_stopping=True,
        max_length=256,
    )[0]['generated_text'])


query("슈카월드에 대해서 알아?")
query("세상에서 가장 유명한 사람은?")
query("알버트 아인슈타인에 대해서 설명해줘")
query("다음 영화에 대해서 설명해줘", "기생충")
query("섭씨 온도를 화씨로 변경해줘", "섭씨 온도: 15도")