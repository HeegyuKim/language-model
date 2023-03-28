MODEL_TYPE="causal-lm"
MODEL_NAME="skt/kogpt2-base-v2"
RUN_NAME="$MODEL_NAME-$DATASET"
PROJECT="gpt2-dialog"

accelerate launch train_torch.py \
    --output_dir "./checkpoint/gpt2-dialog" \
    --project $PROJECT \
    --run_name $RUN_NAME \
    --do_train \
    --do_eval \
    --model_name_or_path $MODEL_NAME \
    --model_type $MODEL_TYPE \
    --task gpt \
    --train_file="/data/dialog-v1-vocab51k-block1024/train/*.jsonl" \
    --validation_file="/data/dialog-v1-vocab51k-block1024/test/*.jsonl" \
    --cache_dir="/data/.cache/" \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps=8 \
    --max_sequence_length 1024 \
    --learning_rate="1e-4" \
    --overwrite_output_dir \
    --num_train_epochs="3" \
    --logging_steps="500" \
    --save_strategy="last" \
    --eval_strategy="epoch"

