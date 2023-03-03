from util import handle_all_files, clean, read_json


def handle_file(file):
    file = read_json(file)
    output = {}
    for item in file.values():
        output.update(item)

    yield {
        'id': output['passage_id'],
        'text': output['passage'],
        'summary': output['summary1'],
    }

if __name__ == "__main__":
    handle_all_files("022.요약문 및 레포트 생성 데이터/01.데이터/1.Training/라벨링데이터", "**/*.json", "output/nia_summary_train.jsonl", handle_file)
    handle_all_files("022.요약문 및 레포트 생성 데이터/01.데이터/2.Validation/라벨링데이터", "**/*.json", "output/nia_summary_validation.jsonl", handle_file)
