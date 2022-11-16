import re
import emoji
from soynlp.normalizer import repeat_normalize, emoticon_normalize
from pathlib import Path
import jsonlines
import json
from tqdm import tqdm
from traceback import print_exc


emojis = "".join(emoji.UNICODE_EMOJI.keys())
pattern = re.compile(f"[^ .,?!/@$%~％·∼()\x00-\x7Fㄱ-힣{emojis}]+")
url_pattern = re.compile(
    r"https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)"
)


def clean(x):
    """
        KcBERT에서 사용된 전처리 방식
        https://huggingface.co/beomi/kcbert-base

        - 정규표현식을 통해 한글, 영어, 특수문자를 포함해 Emoji까지 학습 대상에 포함했습니다.
        - 한편, 한글 범위를 ㄱ-ㅎ가-힣 으로 지정해 ㄱ-힣 내의 한자를 제외했습니다.
        - 댓글 내 중복 문자열 축약: ㅋㅋㅋㅋㅋ와 같이 중복된 글자를 ㅋㅋ와 같은 것으로 합쳤습니다.
    """
    x = pattern.sub(" ", x)
    x = url_pattern.sub("", x)
    x = x.strip()
    x = repeat_normalize(x, num_repeats=2)  # 와하하하하하하하하하핫 -> 와하하핫
    x = emoticon_normalize(x, num_repeats=2)  # ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ쿠ㅜㅜㅜㅜㅜㅜ -> ㅋㅋㅜㅜ
    return x


def join_utterance(uttrs):
    last_s = None
    out = ""
    speakers = []

    for u in uttrs:
        s = u["speaker"]
        t = u["text"]

        if last_s is None:
            out = t
        elif s == last_s:
            out += " " + t
        else:
            out += "\n" + t

        speakers.append(s)
        last_s = s

    return out, speakers


def read_json(filename: str):
    with open(filename, encoding="utf-8") as f:
        return json.load(f)


def iter_jsonlines(filename: str):
    with jsonlines.open(filename) as f:
        yield from f


def handle_all_files(
    dir: str, pattern: str, target_file: str, parse_func, max_items=None
):

    fout = jsonlines.open(target_file, "w")
    i = 0

    for file in tqdm(list(Path(dir).rglob(pattern))):
        try:
            for item in parse_func(file):
                fout.write(item)
                i += 1

                if max_items is not None and i >= max_items:
                    break
            if max_items is not None and i >= max_items:
                break
        except:
            print_exc()
            print("file: ", str(file))

    fout.close()
    print(f"{i} items to {target_file}")
