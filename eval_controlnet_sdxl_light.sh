# Define default values for parameters

# # sdxl light without negative prompt
# export BASE_MODEL="stabilityai/stable-diffusion-xl-base-1.0"
# export REPO="ByteDance/SDXL-Lightning"
# export INFERENCE_STEP=8
# export CKPT="sdxl_lightning_8step_unet.safetensors" # caution!!! ckpt's "N"step must match with inference_step
# export CONTROLNET_MODEL="sdxl_light_custom_caption_output/checkpoint-12500/controlnet"
# export DATASET="nickpai/coco2017-colorization"
# export DATSET_REVISION="custom-caption"
# export OUTPUT_DIR="sdxl_light_custom_caption_output/checkpoint-12500"

# accelerate launch eval_controlnet_sdxl_light.py \
#     --pretrained_model_name_or_path=$BASE_MODEL \
#     --repo=$REPO \
#     --ckpt=$CKPT \
#     --num_inference_steps=$INFERENCE_STEP \
#     --controlnet_model_name_or_path=$CONTROLNET_MODEL \
#     --dataset=$DATASET \
#     --dataset_revision=$DATSET_REVISION \
#     --mixed_precision="fp16" \
#     --output_dir=$OUTPUT_DIR

# sdxl light with negative prompt
export BASE_MODEL="stabilityai/stable-diffusion-xl-base-1.0"
export REPO="ByteDance/SDXL-Lightning"
export INFERENCE_STEP=8
export CKPT="sdxl_lightning_8step_unet.safetensors" # caution!!! ckpt's "N"step must match with inference_step
export CONTROLNET_MODEL="sdxl_light_caption_output/checkpoint-22500/controlnet"
export DATASET="nickpai/coco2017-colorization"
export DATSET_REVISION="custom-caption"
export OUTPUT_DIR="sdxl_light_caption_output/checkpoint-22500"

accelerate launch eval_controlnet_sdxl_light.py \
    --pretrained_model_name_or_path=$BASE_MODEL \
    --repo=$REPO \
    --ckpt=$CKPT \
    --num_inference_steps=$INFERENCE_STEP \
    --controlnet_model_name_or_path=$CONTROLNET_MODEL \
    --dataset=$DATASET \
    --dataset_revision=$DATSET_REVISION \
    --mixed_precision="fp16" \
    --output_dir=$OUTPUT_DIR \
    --negative_prompt