# 데이터 전처리 코드

## NIKL(국립국어원 모두의 말뭉치)

### 구어 말뭉치 버전 1.2
1. `NIKL_SPOKEN_v1.2.zip` 파일을 `nikl_spoken_v1.2` 에 압축 해제
1. 실행 `python nikl_spoken.py` 

### 메신저 말뭉치 버전 2.0
1. `NIKL_MESSENGER_v2.0.zip` 파일을 `nikl_messenger_v2.0`에 압축해제
1. 실행 `python nikl_messenger.py` 

### 신문 말뭉치
1. 세 종류의 신문 말뭉치를받아서 `nikl_news/`에 압축해제<br>
```
- 신문 말뭉치 2021
- 신문 말뭉치 2020
- 신문 말뭉치 v2.0: 2009~2019년 사이의 신문 모음
```
2. 실행 `python nikl_news.py` 


## NIA(AIHub)
### NIA 온라인 구어체 말뭉치 데이터
1. `031.온라인 구어체 말뭉치 데이터/01.데이터/1.Training_220728_add/라벨링데이터/TL1.zip` 을 `nia_spoken` 디렉토리에 압축해제
2. 실행 `python nia_spoken.py`

`게임/BOGA210001274753.json` 파일 31994 라인에 아래처럼 따옴표가 없는 문제가 있어 고쳐야합니다.
```
{
    "id": (반사회적용어),
    "text": "아",
    "tag": "EC"
},
```

### NIA 웹데이터 기반 한국어 말뭉치 데이터
1. `030.웹데이터 기반 한국어 말뭉치 데이터\01.데이터\1.Training\라벨링데이터/TL1.zip` 를 `nia_web/train`에 압축해제
2. `030.웹데이터 기반 한국어 말뭉치 데이터\01.데이터\1.Validation\라벨링데이터/VL1.zip` 를 `nia_web/validation`에 압축해제
3. 실행 `python nia_web.py`

### NIA 대규모 구매도서 기반 한국어 말뭉치 데이터
1. `029.대규모 구매도서 기반 한국어 말뭉치 데이터\01.데이터\4.Sample\sample.zip` 파일 안의 `sample/라벨링데이터/000_DATA.tsv`를 `nia_book`에 저장
2. 실행 `python nia_book.py`

이 데이터는 웹페이지 설명에서는 JSON구조라고 되어있으나 실제로 받아보면 TSV 파일이며, 데이터의 퀄리티도 의심됨.<br/>
317MB 데이터가 필터링을 거치면 1.6MB밖에 안남는다. ㅡㅡ

### NIA 한국어 SNS 데이터
1. `한국어 SNS/Training/한국어SNS_train.zip` 을 `nia_sns/train`에 압축해제
1. `한국어 SNS/Validation/한국어SNS_train.zip` 을 `nia_sns/validation`에 압축해제
1. 실행 `python nia_sns.py`