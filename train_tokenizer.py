from tokenizers import ByteLevelBPETokenizer
import os 

special_tokens = [
    "<pad>",
    "<s>",
    "</s>",
    "<usr>",
    "<sys>",
    "<unk>",
    "<mask>",
    "<|endoftext|>"
] + [f"<unused{i}" for i in range(1, 65)]

tokenizer = ByteLevelBPETokenizer()

dir = "data/tokenizer_data/"
files = [dir + x for x in os.listdir(dir)]
vocab_size = 8000
min_freq = 2

print("train bpe tokenizers from", files)

tokenizer.train(
    files=files,
    vocab_size=vocab_size,
    min_frequency=min_freq,
    special_tokens=special_tokens
)
tokenizer.save_model(f"tokenizer-{vocab_size}/")

