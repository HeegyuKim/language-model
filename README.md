# language-model
TPU에서 한국어 언어모델을 학습하기 위한 코드(jax/flax, pytorch)

### 학습된 모델들
Name | Model | Size | Hidden | # Layers | # Heads | max_seq_len
--- | --- | --- | --- | --- | --- | --- 
[kogpt-j-base](https://huggingface.co/heegyu/kogpt-j-base) | GPT-J | 163M | 768 | 12 | 12 | 1024
[kogpt-j-base-24L](https://huggingface.co/heegyu/kogpt-j-base-24L) | GPT-J  | 237M | 768 | 24 | 12 | 1024
[kogpt-j-350m](https://huggingface.co/heegyu/kogpt-j-350m) | GPT-J | 350M | 1024 | 20 | 16 | 1024

이 프로젝트는 아주대학교 파란학기제와 Google TPU Research Cloud의 지원을 받아서 진행되고 있습니다.

### TODO
- GPT2 125M, 210M(base-24L), 355M
- BART-large


## TPU-VM Setup
클라우드 쉘에서 us-central1-f zone v2-8 잡는 명령어
```
export PROJECT_ID=?
export TPU_NAME=암거나

gcloud config set project ${PROJECT_ID}
gcloud config set account 이메일

gcloud compute tpus tpu-vm create ${TPU_NAME} \
  --zone=us-central1-f \
  --accelerator-type=v2-8 \
  --version=tpu-vm-tf-2.9.2 

# --preemptible 추가하면 최대 24시간 쓸 수 있는데 바로 가능

gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone us-central1-f --project ${PROJECT_ID}
```

버전은 
- tpu-vm-base: 기본, jax 쓰려면 이거
- tpu-vm-tf-2.9.2: tensorflow
- tpu-vm-pt-1.12: 파이토치, 근데 느려터졌음. 내가 잘못했을 수도...

accelerator-type
- v2-8: preemptible 아니어도 잘 잡힘, 실패할 때도 있는데 하다보면 됨
- v3-8: 맨날 남는 거 없다고 실패함 ㅂㄷㅂㄷ zone europe-west4-a 만 가능


### Jax/Flax VM setup
tf vm 으로 한 다음 아래 명령어 쳐서 설치하면 됩니다.
```
python3 -m pip install "jax[tpu]>=0.2.16" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
python3 -m pip install transformers datasets huggingface_hub evaluate accelerate optax flax wandb

export USE_TORCH=False

# 테스트 실행
python3 jax_test.py
```

#### 주의사항
```
# 제일 먼저 선언해야됨, 나중에 선안하면 오류남 ㅡㅡ
import transformers

# 에러남, 이유 모름, 없애버렸음
from transformers.testing_utils import CaptureLogger
```

### Pytorch VM Setup
```
# .bashrc에다가 넣어도 안되고 매번 해줘야한다 이유가 뭘까..
export XRT_TPU_CONFIG="localservice;0;localhost:51011"
export USE_Torch=True

pip3 install transformers wandb datasets tokenizers accelerate hydra-core
```

### 여러 Troubleshoot
#### PERMISSION_DENIED 에러가 발생해요
```
PERMISSION_DENIED: open(/dev/accel0): Operation not permitted: 

# 이미 어떤 프로세스(좀비?)가 tpu를 쓰는지 보고 강제종료한다
> sudo lsof -w /dev/accel0
> kill -9 ?
```


## 학습 방법(GPT)
1. data/generate_for_gpt.py를 실행해서 GPT 학습용 데이터를 tokenizer로 인코딩한 다음 packing한다. 코드 안에서 어떤 huggingface dataset에서 생성할건지 지정한다.
2. packing된 데이터를 불러와서 학습을 진행한다. script/flax/ 안에 있는 shell script 참고

# TODO
- [x] Flax wandb integration
- [x] Flax gradient accumulation
- [ ] Flax resume training
- [ ] Flax CPU Inference
- [ ] Model Parallel