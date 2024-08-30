import os
import time
import torch
import shutil
import argparse
import numpy as np

from tqdm import tqdm
from PIL import Image
from datasets import load_dataset
from accelerate import Accelerator
from diffusers.utils import load_image
from diffusers import (
    AutoencoderKL,
    StableDiffusionXLControlNetPipeline, 
    ControlNetModel,
    UNet2DConditionModel,
)
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

# Define the function to parse arguments
def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a ControlNet evaluation script.")

    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_vae_model_name_or_path",
        type=str,
        default=None,
        help="Path to an improved VAE to stabilize training. For more details check out: https://github.com/huggingface/diffusers/pull/4038.",
    )
    parser.add_argument(
        "--controlnet_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained controlnet model.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        required=True,
        help="Path to output results.",
    )
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="nickpai/coco2017-colorization",
        help="Dataset used"
    )
    parser.add_argument(
        "--dataset_revision", 
        type=str, 
        default="caption-free",
        choices=["main", "caption-free", "custom-caption"],
        help="Revision option (main/caption-free/custom-caption)"
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=8,
        help="1-step, 2-step, 4-step, or 8-step distilled models"
    )
    parser.add_argument(
        "--repo",
        type=str,
        default="ByteDance/SDXL-Lightning",
        required=True,
        help="Repository from huggingface.co",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="sdxl_lightning_4step_unet.safetensors",
        required=True,
        help="Available checkpoints from the repository",
    )
    parser.add_argument(
        "--negative_prompt",
        action="store_true",
        help="The prompt or prompts not to guide the image generation",
    )
    
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args

def apply_color(image, color_map):
    # Convert input images to LAB color space
    image_lab = image.convert('LAB')
    color_map_lab = color_map.convert('LAB')

    # Split LAB channels
    l, a, b = image_lab.split()
    _, a_map, b_map = color_map_lab.split()

    # Merge LAB channels with color map
    merged_lab = Image.merge('LAB', (l, a_map, b_map))

    # Convert merged LAB image back to RGB color space
    result_rgb = merged_lab.convert('RGB')
    
    return result_rgb

def main(args):
    generator = torch.manual_seed(0)

    # Path to the eval_results folder
    eval_results_folder = os.path.join(args.output_dir, "results")

    # Remove eval_results folder if it exists
    if os.path.exists(eval_results_folder):
        shutil.rmtree(eval_results_folder)

    # Create directory for eval_results
    os.makedirs(eval_results_folder)

    # Create subfolders for compare and colorized images
    compare_folder = os.path.join(eval_results_folder, "compare")
    colorized_folder = os.path.join(eval_results_folder, "colorized")
    os.makedirs(compare_folder)
    os.makedirs(colorized_folder)

    # Load the validation split of the colorization dataset
    val_dataset = load_dataset(args.dataset, split="validation", revision=args.dataset_revision)

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
    )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae_path = (
        args.pretrained_model_name_or_path
        if args.pretrained_vae_model_name_or_path is None
        else args.pretrained_vae_model_name_or_path
    )
    vae = AutoencoderKL.from_pretrained(
        vae_path,
        subfolder="vae" if args.pretrained_vae_model_name_or_path is None else None,
        revision=args.revision,
        variant=args.variant,
    )
    unet = UNet2DConditionModel.from_config(
        args.pretrained_model_name_or_path, 
        subfolder="unet", 
        revision=args.revision, 
        variant=args.variant,
    )
    unet.load_state_dict(load_file(hf_hub_download(args.repo, args.ckpt)))

    # Move vae, unet and text_encoder to device and cast to weight_dtype
    # The VAE is in float32 to avoid NaN losses.
    if args.pretrained_vae_model_name_or_path is not None:
        vae.to(accelerator.device, dtype=weight_dtype)
    else:
        vae.to(accelerator.device, dtype=torch.float32)
    unet.to(accelerator.device, dtype=weight_dtype)

    controlnet = ControlNetModel.from_pretrained(args.controlnet_model_name_or_path, torch_dtype=weight_dtype)
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=vae,
        unet=unet,
        controlnet=controlnet, 
    )
    pipe.to(accelerator.device, dtype=weight_dtype)

    # Prepare everything with our `accelerator`.
    pipe, val_dataset = accelerator.prepare(pipe, val_dataset)

    pipe.safety_checker = None

    # Counter for processed images
    processed_images = 0

    # Record start time
    start_time = time.time()

    # Iterate through the validation dataset
    for example in tqdm(val_dataset, desc="Processing Images"):
        image_path = example["file_name"]

        prompt = []
        for caption in example["captions"]:
            if isinstance(caption, str):
                prompt.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                prompt.append(caption[0])
            else:
                raise ValueError(
                    f"Caption column `captions` should contain either strings or lists of strings."
                )
        
        negative_prompt = None    
        if args.negative_prompt:   
            negative_prompt = [
                "low quality, bad quality, low contrast, black and white, bw, monochrome, grainy, blurry, historical, restored, desaturate"
            ]

        # Generate image
        ground_truth_image = load_image(image_path).resize((512, 512))
        control_image = load_image(image_path).convert("L").convert("RGB").resize((512, 512))
        image = pipe(prompt=prompt, 
                     negative_prompt=negative_prompt, 
                     num_inference_steps=args.num_inference_steps, 
                     generator=generator, 
                     image=control_image).images[0]

        # Apply color mapping
        image = apply_color(ground_truth_image, image)
        
        # Concatenate images into a row
        row_image = np.hstack((np.array(control_image), np.array(image), np.array(ground_truth_image)))
        row_image = Image.fromarray(row_image)

        # Save row image in the compare folder
        compare_output_path = os.path.join(compare_folder, f"{image_path.split('/')[-1]}")
        row_image.save(compare_output_path)

        # Save colorized image in the colorized folder
        colorized_output_path = os.path.join(colorized_folder, f"{image_path.split('/')[-1]}")
        image.save(colorized_output_path)

        # Increment processed images counter
        processed_images += 1

    # Record end time
    end_time = time.time()

    # Calculate total time taken
    total_time = end_time - start_time

    # Calculate FPS
    fps = processed_images / total_time

    print("All images processed.")
    print(f"Total time taken: {total_time:.2f} seconds")
    print(f"FPS: {fps:.2f}")

# Entry point of the script
if __name__ == "__main__":
    args = parse_args()
    main(args)