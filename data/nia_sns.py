from util import handle_all_files, clean, read_json, join_utterance
import ijson


def handle_file(file):
    f = open(str(file), encoding="utf-8")
    for items in ijson.items(f, "data"):
        for item in items:
            uttrs = []

            for uttr in item["body"]:
                s = uttr["participantID"]

                if s[1:].isnumeric():
                    s = int(s[1:])
                else:
                    s = -1

                u = uttr["utterance"]

                uttrs.append({"speaker": s, "text": u})

            text, speakers = join_utterance(uttrs)

            yield {"text": clean(text), "speakers": speakers}

    f.close()


if __name__ == "__main__":
    handle_all_files("nia_sns/", "**/*.json", "output/nia_sns.jsonl", handle_file)
