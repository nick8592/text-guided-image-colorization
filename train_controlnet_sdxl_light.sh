# Original ControlNet paper: 
# "In the training process, we randomly replace 50% text prompts ct with empty strings. 
# This approach increases ControlNet’s ability to directly recognize semantics 
# in the input conditioning images (e.g., edges, poses, depth, etc.) as a replacement for the prompt."
# https://civitai.com/articles/2078/play-in-control-controlnet-training-setup-guide

export MODEL_DIR="stabilityai/stable-diffusion-xl-base-1.0"
export REPO="ByteDance/SDXL-Lightning"
export INFERENCE_STEP=8
export CKPT="sdxl_lightning_8step_unet.safetensors" # caution!!! ckpt's "N"step must match with inference_step
export OUTPUT_DIR="test"
export PROJECT_NAME="train_sdxl_light_controlnet"
export DATASET="nickpai/coco2017-colorization"
export REVISION="custom-caption" # option: main/caption-free/custom-caption
export VAL_IMG_NAME="'./000000295478.jpg' './000000122962.jpg' './000000000285.jpg' './000000007991.jpg' './000000018837.jpg' './000000000724.jpg'"
export VAL_PROMPT="'Woman walking a small dog behind her.' 'A group of children sitting at a long table eating pizza.' 'A close up picture of a bear face.' 'A plate on a table is filled with carrots and beans.' 'A large truck on a city street with two works sitting on top and one worker climbing in through door.' 'An upside down stop sign by the road.'"
# export VAL_PROMPT="'Colorize this image as if it was taken with a color camera' 'Colorize this image' 'Add colors to this image' 'Make this image colorful' 'Colorize this grayscale image' 'Add colors to this image'"

accelerate launch train_controlnet_sdxl_light.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --tracker_project_name=$PROJECT_NAME \
 --seed=123123 \
 --dataset_name=$DATASET \
 --dataset_revision=$REVISION \
 --image_column="file_name" \
 --conditioning_image_column="file_name" \
 --caption_column="captions" \
 --max_train_samples=100000 \
 --num_validation_images=1 \
 --resolution=512 \
 --num_train_epochs=5 \
 --dataloader_num_workers=8 \
 --learning_rate=1e-5 \
 --train_batch_size=2 \
 --gradient_accumulation_steps=8 \
 --proportion_empty_prompts=0 \
 --validation_steps=500 \
 --checkpointing_steps=2500 \
 --mixed_precision="fp16" \
 --gradient_checkpointing \
 --use_8bit_adam \
 --repo=$REPO \
 --ckpt=$CKPT \
 --num_inference_steps=$INFERENCE_STEP \
 --enable_xformers_memory_efficient_attention

#  --validation_image './000000295478.jpg' './000000122962.jpg' './000000000285.jpg' './000000007991.jpg' './000000018837.jpg' './000000000724.jpg' \
#  --validation_prompt 'Woman walking a small dog behind her.' 'A group of children sitting at a long table eating pizza.' 'A close up picture of a bear face.' 'A plate on a table is filled with carrots and beans.' 'A large truck on a city street with two works sitting on top and one worker climbing in through door.' 'An upside down stop sign by the road.' \
 