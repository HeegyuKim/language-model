from util import handle_all_files, clean, read_json, join_utterance


def handle_file(file):
    file = read_json(file)
    users = []

    for ne in file["document"]:
        uttrs = []

        for uttr in ne["utterance"]:
            s = uttr["speaker_id"]
            if s not in users:
                users.append(s)
            s = users.index(s)
            u = clean(uttr["form"])

            uttrs.append({"speaker": s, "text": u})

        text, speakers = join_utterance(uttrs)
        uttrs = text.split("\n")

        lines = []
        for s, u in zip(speakers, uttrs):
            lines.append(f"{s}: {u}")
        

        yield {"dialog": "\n".join(lines), "depth": len(lines), "id": "nikl-dialog-2021-v1.0-" + file["id"]}


if __name__ == "__main__":
    handle_all_files(
        "nikl_dialogue_2021_v1.0/",
        "**/*.json",
        "output/nikl_dialogue_2021_v1.0.jsonl",
        handle_file,
    )
