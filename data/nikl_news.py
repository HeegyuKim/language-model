from util import handle_all_files, clean, read_json, join_utterance


def handle_file(file):
    file = read_json(file)

    for doc in file["document"]:
        item = doc["metadata"].copy()
        para = doc["paragraph"]
        paras = [clean(p["form"]) for p in para]

        item["text"] = "\n".join(paras)

        yield item


if __name__ == "__main__":
    dirs = [
        # ("국립국어원 신문 말뭉치(버전 2.0)", "nikl_news_v2.0.jsonl"),
        # ("국립국어원 신문 말뭉치 2020(버전 1.1)", "nikl_news_2020_v1.1.jsonl"),
        ("NIKL_NEWSPAPER_2021_v1.0", "nikl_news_2021_v1.0.jsonl"),
    ]

    for d, f in dirs:
        handle_all_files("nikl_news", f"{d}/**/*.json", f"output/{f}", handle_file)
