# language-model

## TPU-VM 쓰려면
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
- tpu-vm-pt-1.12: 파이토치, 근데 느려터졌음

accelerator-type
- v2-8: preemptible 아니어도 잘 잡힘, 실패할 때도 있는데 하다보면 됨
- v3-8: 맨날 남는 거 없다고 실패함 ㅂㄷㅂㄷ zone europe-west4-a 만 가능


## TPU-VM 에서 jax/flax 학습하기
```
python3 -m pip install "jax[tpu]>=0.2.16" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
python3 -m pip install transformers datasets huggingface_hub evaluate accelerate optax flax

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

## TPU-VM에서 Pytorch 학습하기
WIP...
```
export XRT_TPU_CONFIG="localservice;0;localhost:51011"
```


# TODO
- [ ] wandb integration
- [ ] gradient accumulation
