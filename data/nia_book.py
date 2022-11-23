import pandas as pd
from util import clean, get_korean_ratio


def get_doc_id(id):
    return id.split(".")[1]

def main():
    cols = [str(i) for i in range(7)]
    data = pd.read_csv("nia_book/000_DATA.tsv", sep="\t", names=cols)
    print(data.head())

    data["doc_id"] = data["0"].map(get_doc_id)
    data["text"] = data[["doc_id", "1"]].groupby("doc_id")["1"].transform(lambda x: clean("\n".join([str(i) for i in x])))
    data = data[["doc_id", "text"]].drop_duplicates()
    data = data[data.text.str.len() >= 256] # 짧은거 제거
    
    kor_ratio = data.text.map(get_korean_ratio)
    data = data[kor_ratio > 0.75] # 한국어 비중이 75% 이상인 것만 남긴다, 무슨 소스코드랑 에러 메시지같이 이상한게 섞여있다.
    print(data.head())

    data.to_json("output/nia_book.jsonl", orient="records", lines=True, force_ascii=False)


if __name__ == "__main__":
    main()