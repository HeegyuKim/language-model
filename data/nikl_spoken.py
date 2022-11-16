from util import handle_all_files, clean, read_json, join_utterance



def handle_file(file):
    file = read_json(file)

    for ne in file["document"]:
        uttrs = []

        for uttr in ne["utterance"]:
            s = uttr["speaker_id"]

            if s[1:].isnumeric():
                s = int(s[1:])
            else:
                s = -1

            u = uttr["form"]
            if len(u) == 0:
                u = uttr["note"]

            uttrs.append({
                "speaker": s,
                "text": u
            })
        
        text, speakers = join_utterance(uttrs)

        yield {
            "text": text,
            "speakers": speakers
        }

if __name__ == "__main__":
    handle_all_files(
        "nikl_spoken_v1.2",
        "**/*.json",
        "output/nikl_spoken_v1.2.jsonl",
        handle_file
    )