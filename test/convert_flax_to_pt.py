from transformers import AutoModelForCausalLM, GPT2LMHeadModel, GPTJForCausalLM

path = "../../huggingface/kogpt-j-350m/"

model = GPTJForCausalLM.from_pretrained(
    path,
    from_flax=True
)

model.save_pretrained(path)