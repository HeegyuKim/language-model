from util import handle_all_files, clean, read_json, join_utterance


def handle_file(file):
    file = read_json(file)

    for doc in file["document"]:
        item = doc["metadata"].copy()
        para = doc["paragraph"]
        paras = [p["form"] for p in para]
        
        item["text"] = "\n".join(paras)

        yield item



if __name__ == "__main__":
    handle_all_files(
        "nikl_written_v1.2", "**/*.json", "output/nikl_written_v1.2.jsonl", handle_file
    )
