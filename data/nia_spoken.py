from util import handle_all_files, clean, read_json


def handle_file(file):
    file = read_json(file)

    for ne in file["named_entity"]:
        sents = [x["sentence"] for x in ne["content"]]
        text = " ".join(sents)
        text = clean(text)

        yield {"text": text}


if __name__ == "__main__":
    handle_all_files("nia_spoken/", "**/*.json", "output/nia_spoken.jsonl", handle_file)
