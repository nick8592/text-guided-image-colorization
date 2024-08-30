import os
import time
import torch
import shutil
import argparse
import numpy as np

from tqdm import tqdm
from PIL import Image
from datasets import load_dataset
from diffusers.utils import load_image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel

# Define the function to parse arguments
def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a ControlNet evaluation script.")

    parser.add_argument("--model_dir", type=str, default="sd_v2_caption_free_output/checkpoint-22500",
                        help="Directory of the model checkpoint")
    parser.add_argument("--model_id", type=str, default="stabilityai/stable-diffusion-2-base",
                        help="ID of the model (Tested with runwayml/stable-diffusion-v1-5 and stabilityai/stable-diffusion-2-base)")
    parser.add_argument("--dataset", type=str, default="nickpai/coco2017-colorization",
                        help="Dataset used")
    parser.add_argument("--revision", type=str, default="caption-free",
                        choices=["main", "caption-free"],
                        help="Revision option (main/caption-free)")
    
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

    # MODEL_DIR = "sd_v2_caption_free_output/checkpoint-22500"
    # # MODEL_ID="runwayml/stable-diffusion-v1-5"
    # MODEL_ID="stabilityai/stable-diffusion-2-base"
    # DATASET = "nickpai/coco2017-colorization"
    # REVISION = "caption-free" # option: main/caption-free

    # Path to the eval_results folder
    eval_results_folder = os.path.join(args.model_dir, "results")

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
    val_dataset = load_dataset(args.dataset, split="validation", revision=args.revision)

    controlnet = ControlNetModel.from_pretrained(f"{args.model_dir}/controlnet", torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        args.model_id, controlnet=controlnet, torch_dtype=torch.float16
    ).to("cuda")

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

        # Generate image
        ground_truth_image = load_image(image_path).resize((512, 512))
        control_image = load_image(image_path).convert("L").convert("RGB").resize((512, 512))
        image = pipe(prompt, num_inference_steps=20, generator=generator, image=control_image).images[0]

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