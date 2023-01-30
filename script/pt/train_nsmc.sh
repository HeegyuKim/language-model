export USE_TORCH=True

MODEL_NAME="heegyu/kogpt-j-base"
MODEL_TYPE="sequence-classification"
RUN_NAME="nsmc-$MODEL_NAME"
PROJECT="nsmc"

# --do_train \
accelerate launch train_classifier.py \
    --output_dir ./checkpoint/$RUN_NAME \
    --project $PROJECT\
    --run_name RUN_NAME \
    --do_eval \
    --num_procs 8 \
    --model_name_or_path $MODEL_NAME \
    --model_type $MODEL_TYPE \
    --dataset_name nsmc \
    --num_train_epochs 2 \
    --num_labels 2