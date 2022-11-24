from util import handle_all_files, clean, read_json, join_utterance


def handle_file(file):
    file = read_json(file)

    for ne in file["document"]:
        uttrs = []

        for uttr in ne["utterance"]:
            s = int(uttr["speaker_id"])
            u = clean(uttr["form"])

            uttrs.append({"speaker": s, "text": u})

        text, speakers = join_utterance(uttrs)

        yield {"text": text, "speakers": speakers}


if __name__ == "__main__":
    handle_all_files(
        "nikl_messenger_v2.0",
        "**/*.json",
        "output/nikl_messenger_v2.0.jsonl",
        handle_file,
    )
