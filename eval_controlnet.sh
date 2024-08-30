# Define default values for parameters

# # sdv2 with BCE loss
# MODEL_DIR="sd_v2_caption_bce_output/checkpoint-22500"
# MODEL_ID="stabilityai/stable-diffusion-2-base"
# DATASET="nickpai/coco2017-colorization"
# REVISION="main"

# sdv2 with kl loss
MODEL_DIR="sd_v2_caption_kl_output/checkpoint-22500"
MODEL_ID="stabilityai/stable-diffusion-2-base"
DATASET="nickpai/coco2017-colorization"
REVISION="main"

accelerate launch eval_controlnet.py \
    --model_dir=$MODEL_DIR \
    --model_id=$MODEL_ID \
    --dataset=$DATASET \
    --revision=$REVISION