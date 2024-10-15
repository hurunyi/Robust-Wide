# Script Name: train.sh
# Description: launch the training process
# Author: Runyi Hu
# Date: 2024-10-15

OUTPUT_DIR="./train_results"
DATA_DIR="./data"
WM_MODEL_CONFIG="./config.yaml"

#######################DEFAULT_SETTING#######################
IMAGE_SIZE=512
LEARNING_RATE=1e-3
BATCH_SIZE=2
GRADIENT_ACCUMULATION_STEPS=1
MAX_TRAIN_STEPS=20000
LR_WARMUP_STEPS=400
DECODER_WEIGHT=0.1
ENC_LATENT_LOSS=0.001
LAST_GRAD_STEPS=3
#######################BEST_SETTING############################

accelerate config

accelerate launch train.py \
  --train_data_dir $DATA_DIR \
  --wm_model_config $WM_MODEL_CONFIG \
  --output_dir $OUTPUT_DIR \
  --image_size $IMAGE_SIZE \
  --batch_size $BATCH_SIZE \
  --max_train_steps $MAX_TRAIN_STEPS \
  --learning_rate $LEARNING_RATE \
  --lr_scheduler "cosine" \
  --lr_warmup_steps $LR_WARMUP_STEPS \
  --log_steps 10 \
  --save_steps 1 \
  --decoder_weight $DECODER_WEIGHT \
  --last_grad_steps $LAST_GRAD_STEPS \
  --enc_latent_weight $ENC_LATENT_LOSS \
  --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS
