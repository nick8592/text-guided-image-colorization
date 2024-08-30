# sdxl light for single image
export BASE_MODEL="stabilityai/stable-diffusion-xl-base-1.0"
export REPO="ByteDance/SDXL-Lightning"
export INFERENCE_STEP=8
export CKPT="sdxl_lightning_8step_unet.safetensors" # caution!!! ckpt's "N"step must match with inference_step
export CONTROLNET_MODEL="sdxl_light_caption_output/checkpoint-30000/controlnet"
export CAPTION_MODEL="blip-image-captioning-large"
export IMAGE_PATH="example/legacy_images/Hollywood-Sign.jpg"
# export POSITIVE_PROMPT="blue shirt"

accelerate launch eval_controlnet_sdxl_light_single.py \
    --pretrained_model_name_or_path=$BASE_MODEL \
    --repo=$REPO \
    --ckpt=$CKPT \
    --num_inference_steps=$INFERENCE_STEP \
    --controlnet_model_name_or_path=$CONTROLNET_MODEL \
    --caption_model_name=$CAPTION_MODEL \
    --mixed_precision="fp16" \
    --image_path=$IMAGE_PATH \
    --positive_prompt="red car"