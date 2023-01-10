from transformers import AutoModelForCausalLM

path = "../huggingface/kogpt-j-350m/"

model = AutoModelForCausalLM.from_pretrained(
    path,
    from_flax=True
)

model.save_pretrained(path)