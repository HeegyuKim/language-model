from transformers import AutoModelForCausalLM, GPT2LMHeadModel, GPTJForCausalLM

path = "../../huggingface/ajoublue-gpt2-medium/"

model = GPT2LMHeadModel.from_pretrained(
    path,
    from_flax=True
)

model.save_pretrained(path)