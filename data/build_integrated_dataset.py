from datasets import load_dataset


dataset = load_dataset("json", data_dir="test/")

print(dataset)
print(dataset["train"][0])
print(dataset["train"][1])
