from util import handle_all_files, clean, read_json


def handle_file(file):
    file = read_json(file)

    for ne in file["named_entity"]:
        sents = [x["sentence"] for x in ne["content"]]

        yield {"title": ne["title"][0]["sentence"], "text": " ".join(sents)}


if __name__ == "__main__":
    handle_all_files("nia_web/", "**/*.json", "output/nia_web.jsonl", handle_file)
