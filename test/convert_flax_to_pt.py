from transformers import AutoModelForCausalLM, GPT2LMHeadModel, FlaxGPT2LMHeadModel

path = "../huggingface/ajoublue-gpt2-base-24L/"

model = GPT2LMHeadModel.from_pretrained(
    path,
    from_flax=True
)
# model = FlaxGPT2LMHeadModel.from_pretrained(
#     path
# )

model.save_pretrained(path)