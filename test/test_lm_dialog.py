from transformers import pipeline

print('start model loading')
generator = pipeline(
    'text-generation',
    # model='/data2/checkpoint/gpt-j-base-dialog/epoch-7/',
    model='/data2/checkpoint/gpt-j-350m-dialog/epoch-9/',
    tokenizer='heegyu/kogpt-j-base'
)

print('start test generation')
generation_args = dict(
    repetition_penalty=1.3,
    no_repeat_ngram_size=4,
    eos_token_id=375, # \n
    max_new_tokens=32,
    do_sample=True,
    top_p=0.7,
    early_stopping=True
)
print(generator(
    ["0 : **는 게임 좋아하니\n1 :",
    "0 : 어제 강남에서 살인사건 났대 ㅜㅜ 너무 무서워\n1 : 헐 왜? 무슨 일 있었어?\n0 : 사진보니까 막 피흘리는 사람있고 경찰들이 떠서 제압하고 난리도 아니었다던데??\n1 :",
    "0 : 자기야 어제는 나한테 왜 그랬어?\n1 : 뭔 일 있었어?\n0 : 어떻게 나한테 말도 없이 그럴 수 있어? 나 진짜 실망했어\n1 : "],
    do_sample=True,
    max_new_tokens=32
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
