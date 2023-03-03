MODEL_TYPE="director"
RUN_NAME="director-gpt2"
PROJECT="director"

# MODEL_NAME="gpt2"
# DATASET="hate_speech18"
# EVAL_TOXIC_CLASSIFIER="martin-ha/toxic-comment-model"

MODEL_NAME="skt/kogpt2-base-v2"
DATASET="jason9693/APEACH"
EVAL_TOXIC_CLASSIFIER="jason9693/koelectra-base-v3-discriminator-apeach"

accelerate launch train_torch.py \
    --output_dir "./checkpoint/$RUN_NAME" \
    --project $PROJECT \
    --run_name "$RUN_NAME-frozen" \
    --do_eval --do_train \
    --model_name_or_path $MODEL_NAME \
    --model_type $MODEL_TYPE \
    --dataset_name $DATASET \
    --task director \
    --num_train_epochs 50 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 5e-5 \
    --max_sequence_length 128 \
    --save_strategy last \
    --logging_steps 50 \
    --director_gamma_train 1.0 \
    --director_gamma_generate 5 \
    --director_eval_classifier $EVAL_TOXIC_CLASSIFIER \
    --director_frozen false

accelerate launch train_torch.py \
    --output_dir "./checkpoint/$RUN_NAME" \
    --project $PROJECT \
    --run_name "$RUN_NAME-frozen" \
    --do_eval --do_train \
    --model_name_or_path $MODEL_NAME \
    --model_type $MODEL_TYPE \
    --dataset_name $DATASET \
    --task director \
    --num_train_epochs 50 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 5e-5 \
    --max_sequence_length 128 \
    --save_strategy last \
    --logging_steps 50 \
    --director_gamma_train 0.5 \
    --director_gamma_generate 5 \
    --director_eval_classifier $EVAL_TOXIC_CLASSIFIER \
    --director_frozen false

accelerate launch train_torch.py \
    --output_dir "./checkpoint/$RUN_NAME" \
    --project $PROJECT \
    --run_name "$RUN_NAME-frozen" \
    --do_eval --do_train \
    --model_name_or_path $MODEL_NAME \
    --model_type $MODEL_TYPE \
    --dataset_name $DATASET \
    --task director \
    --num_train_epochs 50 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps=8 \
    --learning_rate 5e-5 \
    --max_sequence_length 32 \
    --save_strategy last \
    --logging_steps 50 \
    --director_gamma_train 2.5 \
    --director_gamma_generate 5 \
    --director_eval_classifier $EVAL_TOXIC_CLASSIFIER \
    --director_frozen true

# accelerate launch train_torch.py \
#     --output_dir "./checkpoint/$RUN_NAME" \
#     --project $PROJECT \
#     --run_name "$RUN_NAME-frozen" \
#     --do_eval --do_train \
#     --model_name_or_path $MODEL_NAME \
#     --model_type $MODEL_TYPE \
#     --dataset_name $DATASET \
#     --task director \
#     --num_train_epochs 50 \
#     --per_device_train_batch_size 1 \
#     --per_device_eval_batch_size 1 \
#     --learning_rate 5e-5 \
#     --max_sequence_length 128 \
#     --save_strategy last \
#     --logging_steps 50 \
#     --director_gamma_train 1.0 \
#     --director_gamma_generate 5 \
#     --director_eval_classifier $EVAL_TOXIC_CLASSIFIER \
#     --director_frozen true