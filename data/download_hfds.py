from datasets import load_dataset
import os



datasets = [
    # "heegyu/kowikitext",
    # "heegyu/namuwiki-extracted",
    # "heegyu/aihub_sns_dialog_gpt",
    # "heegyu/nikl_messenger_dialog_gpt",
    "heegyu/aihub_spoken_2021"
]


def load_dataset_renamed(name):
    ds = load_dataset(name, split="train", use_auth_token=True)
    name_vars = ["sentence", "dialog", "spoken"]

    for var in name_vars:
        if var in ds.column_names:
            ds = ds.rename_column(var, "text")
            break

    columns = set(ds.column_names)
    columns.remove("text")
    ds = ds.remove_columns(columns)
    ds = ds.filter(lambda x: len(x["text"]) >= 8)

    return ds

target_dir = "test"
os.makedirs(target_dir, exist_ok=True)

for name in datasets:
    dataset = load_dataset_renamed(name)
    name = name.replace("/", "__")
    dataset.to_json(f"{target_dir}/{name}.jsonl", lines=True, force_ascii=False)