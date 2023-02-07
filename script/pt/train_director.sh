MODEL_TYPE="director"
MODEL_NAME="gpt2"
RUN_NAME="director-gpt2"
PROJECT="director"

accelerate launch train_torch.py \
    --output_dir "./checkpoint/$RUN_NAME" \
    --project $PROJECT \
    --run_name "$RUN_NAME-frozen" \
    --do_eval --do_train \
    --model_name_or_path $MODEL_NAME \
    --model_type $MODEL_TYPE \
    --task director \
    --num_train_epochs 100 \
    --max_sequence_length 128 \
    --save_strategy last \
    --logging_steps 50 \
    --director_gamma_train 0.2 \
    --director_gamma_generate 5 \
    --director_frozen true

accelerate launch train_torch.py \
    --output_dir "./checkpoint/$RUN_NAME" \
    --project $PROJECT \
    --run_name $RUN_NAME \
    --do_eval --do_train \
    --model_name_or_path $MODEL_NAME \
    --model_type $MODEL_TYPE \
    --task director \
    --num_train_epochs 100 \
    --max_sequence_length 128 \
    --save_strategy last \
    --logging_steps 50 \
    --director_gamma_train 0.2 \
    --director_gamma_generate 5 \
    --director_frozen false