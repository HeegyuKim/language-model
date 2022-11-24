from transformers import GPTNeoXTokenizerFast, GPT2Tokenizer, AutoTokenizer


src_name = "models/tokenizer-51k"
tgt_name = "models/kogpt-neox-base"

tokenizer = GPT2Tokenizer.from_pretrained(src_name)
tokenizer.add_special_tokens({
    "bos_token": "<s>",
    "eos_token": "</s>",
    "pad_token": "<pad>",
    "unk_token": "<unk>",
})
print(tokenizer)

tokenizer.save_pretrained(tgt_name)

# fast = GPTNeoXTokenizerFast(
#     vocab_file=f"{name}/vocab.json",
#     merges_file=f"{name}/merges.txt",
#     bos_token="<s>",
#     eos_token="</s>",
#     pad_token="<pad>",
#     unk_token="<unk>",
# )
fast = AutoTokenizer.from_pretrained(tgt_name)
print(fast)

text = "안녕 ㅋㅋ 하세요. 오늘 뭐 하시나요? 즐거운 하루가 됬으면 좋겠습니다. ㅎㅎ;;; ㅋㅋ 😂 ㅎ"
ids = fast.encode(text, add_special_tokens=True)
print(ids)
print(fast.tokenize(text))
print(fast.decode(ids))
