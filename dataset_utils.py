import json
from datasets import load_dataset, interleave_datasets, DatasetDict, concatenate_datasets


def normalize_prob(x):
    s = sum(x)

    return [i / s for i in x]
    
def load_dataset_renamed(name, validation_count=1000, cache_dir=None, use_auth_token=True):
    dataset = load_dataset(name)
    name_vars = ["sentence", "dialog", "spoken"]

    for split, ds in dataset.items():
        for var in name_vars:
            if var in ds.column_names:
                ds = ds.rename_column(var, "text")

        columns = set(ds.column_names)
        columns.remove("text")
        ds = ds.remove_columns(columns)
        dataset[split] = ds

    return dataset

def interleave_datasets_from_json(file):
    with open(file) as f:
        datasets = json.load(f)
    
    probs = normalize_prob(datasets.values())
    datasets = [load_dataset_renamed(x) for x in datasets.keys()]
    

    out = DatasetDict()
    for split in ["train", "validation"]:
        # out[split] = interleave_datasets([d[split] for d in datasets], probs, stopping_strategy="all_exhausted")
        out[split] = concatenate_datasets([d[split] for d in datasets])
    return out