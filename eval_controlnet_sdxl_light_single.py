import os
import PIL
import time
import torch
import argparse

from typing import Optional, Union
from accelerate import Accelerator
from diffusers import (
    AutoencoderKL,
    StableDiffusionXLControlNetPipeline, 
    ControlNetModel,
    UNet2DConditionModel,
)
from transformers import (
    BlipProcessor, BlipForConditionalGeneration, 
    VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
)
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

# Define the function to parse arguments
def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a ControlNet evaluation script.")
    parser.add_argument(
        "--image_path",
        type=str,
        default="example/legacy_images/Hollywood-Sign.jpg",
        required=True,
        help="Path to the image",
    )
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
        "--caption_model_name",
        type=str,
        default="blip-image-captioning-large",
        choices=["blip-image-captioning-large", "blip-image-captioning-base"],
        help="Path to pretrained controlnet model.",
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
        "--seed",
        type=int,
        default=123,
        help="Random seeds"
    )
    parser.add_argument(
        "--positive_prompt",
        type=str,
        help="Text for positive prompt",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="low quality, bad quality, low contrast, black and white, bw, monochrome, grainy, blurry, historical, restored, desaturate",
        help="Text for negative prompt",
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
    merged_lab = PIL.Image.merge('LAB', (l, a_map, b_map))

    # Convert merged LAB image back to RGB color space
    result_rgb = merged_lab.convert('RGB')
    
    return result_rgb

def remove_unlikely_words(prompt: str) -> str:
    """
    Removes unlikely words from a prompt.

    Args:
        prompt: The text prompt to be cleaned.

    Returns:
        The cleaned prompt with unlikely words removed.
    """
    unlikely_words = []

    a1_list = [f'{i}s' for i in range(1900, 2000)]
    a2_list = [f'{i}' for i in range(1900, 2000)]
    a3_list = [f'year {i}' for i in range(1900, 2000)]
    a4_list = [f'circa {i}' for i in range(1900, 2000)]
    b1_list = [f"{year[0]} {year[1]} {year[2]} {year[3]} s" for year in a1_list]
    b2_list = [f"{year[0]} {year[1]} {year[2]} {year[3]}" for year in a1_list]
    b3_list = [f"year {year[0]} {year[1]} {year[2]} {year[3]}" for year in a1_list]
    b4_list = [f"circa {year[0]} {year[1]} {year[2]} {year[3]}" for year in a1_list]

    words_list = [
        "black and white,", "black and white", "black & white,", "black & white", "circa", 
        "balck and white,", "monochrome,", "black-and-white,", "black-and-white photography,", 
        "black - and - white photography,", "monochrome bw,", "black white,", "black an white,",
        "grainy footage,", "grainy footage", "grainy photo,", "grainy photo", "b&w photo",
        "back and white", "back and white,", "monochrome contrast", "monochrome", "grainy",
        "grainy photograph,", "grainy photograph", "low contrast,", "low contrast", "b & w",
        "grainy black-and-white photo,", "bw", "bw,",  "grainy black-and-white photo",
        "b & w,", "b&w,", "b&w!,", "b&w", "black - and - white,", "bw photo,", "grainy  photo,",
        "black-and-white photo,", "black-and-white photo", "black - and - white photography",
        "b&w photo,", "monochromatic photo,", "grainy monochrome photo,", "monochromatic",
        "blurry photo,", "blurry,", "blurry photography,", "monochromatic photo",
        "black - and - white photograph,", "black - and - white photograph", "black on white,",
        "black on white", "black-and-white", "historical image,", "historical picture,", 
        "historical photo,", "historical photograph,", "archival photo,", "taken in the early",
        "taken in the late", "taken in the", "historic photograph,", "restored,", "restored", 
        "historical photo", "historical setting,",
        "historic photo,", "historic", "desaturated!!,", "desaturated!,", "desaturated,", "desaturated", 
        "taken in", "shot on leica", "shot on leica sl2", "sl2",
        "taken with a leica camera", "taken with a leica camera", "leica sl2", "leica", "setting", 
        "overcast day", "overcast weather", "slight overcast", "overcast", 
        "picture taken in", "photo taken in", 
        ", photo", ",  photo", ",   photo", ",    photo", ", photograph",
        ",,", ",,,", ",,,,", " ,", "  ,", "   ,", "    ,", 
    ]

    unlikely_words.extend(a1_list)
    unlikely_words.extend(a2_list)
    unlikely_words.extend(a3_list)
    unlikely_words.extend(a4_list)
    unlikely_words.extend(b1_list)
    unlikely_words.extend(b2_list)
    unlikely_words.extend(b3_list)
    unlikely_words.extend(b4_list)
    unlikely_words.extend(words_list)
    
    for word in unlikely_words:
        prompt = prompt.replace(word, "")
    return prompt

def blip_image_captioning(image: PIL.Image.Image,
                          model_backbone: str,
                          weight_dtype: type,
                          device: str,
                          conditional: bool) -> str:
    # https://huggingface.co/Salesforce/blip-image-captioning-large
    # https://huggingface.co/Salesforce/blip-image-captioning-base
    if weight_dtype == torch.bfloat16: # in case model might not accept bfloat16 data type
        weight_dtype = torch.float16

    processor = BlipProcessor.from_pretrained(f"Salesforce/{model_backbone}")
    model = BlipForConditionalGeneration.from_pretrained(
         f"Salesforce/{model_backbone}", torch_dtype=weight_dtype).to(device)
    
    valid_backbones = ["blip-image-captioning-large", "blip-image-captioning-base"]
    if model_backbone not in valid_backbones:
         raise ValueError(f"Invalid model backbone '{model_backbone}'. \
                          Valid options are: {', '.join(valid_backbones)}")

    if conditional:
        text = "a photography of"
        inputs = processor(image, text, return_tensors="pt").to(device, weight_dtype)
    else:
        inputs = processor(image, return_tensors="pt").to(device)
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

import matplotlib.pyplot as plt

def display_images(input_image, output_image, ground_truth):
    """
    Displays a grid of input, output, ground truth images with a caption at the bottom.

    Args:
        input_image: A grayscale image as a NumPy array.
        output_image: A grayscale image (result) as a NumPy array.
        ground_truth: A grayscale image (ground truth) as a NumPy array.
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 8))

    axes[0].imshow(input_image, cmap='gray')
    axes[0].set_title('Input')
    axes[0].axis('off')

    axes[1].imshow(output_image)
    axes[1].set_title('Output')
    axes[1].axis('off')

    axes[2].imshow(ground_truth)
    axes[2].set_title('Ground Truth')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

# Define a function to process the image with the loaded model
def process_image(image_path: str,
                  controlnet_model_name_or_path: str,
                  caption_model_name: str,
                  positive_prompt: Optional[str],
                  negative_prompt: Optional[str],
                  seed: int,
                  num_inference_steps: int,
                  mixed_precision: str,
                  pretrained_model_name_or_path: str,
                  pretrained_vae_model_name_or_path: Optional[str],
                  revision: Optional[str],
                  variant: Optional[str],
                  repo: str,
                  ckpt: str,) -> PIL.Image.Image:
    # Seed
    generator = torch.manual_seed(seed)

    # Accelerator Setting
    accelerator = Accelerator(
        mixed_precision=mixed_precision,
    )
    
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae_path = (
        pretrained_model_name_or_path
        if pretrained_vae_model_name_or_path is None
        else pretrained_vae_model_name_or_path
    )
    vae = AutoencoderKL.from_pretrained(
        vae_path,
        subfolder="vae" if pretrained_vae_model_name_or_path is None else None,
        revision=revision,
        variant=variant,
    )
    unet = UNet2DConditionModel.from_config(
        pretrained_model_name_or_path, 
        subfolder="unet", 
        revision=revision, 
        variant=variant,
    )
    unet.load_state_dict(load_file(hf_hub_download(repo, ckpt)))

    # Move vae, unet and text_encoder to device and cast to weight_dtype
    # The VAE is in float32 to avoid NaN losses.
    if pretrained_vae_model_name_or_path is not None:
        vae.to(accelerator.device, dtype=weight_dtype)
    else:
        vae.to(accelerator.device, dtype=torch.float32)
    unet.to(accelerator.device, dtype=weight_dtype)

    controlnet = ControlNetModel.from_pretrained(controlnet_model_name_or_path, torch_dtype=weight_dtype)
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        pretrained_model_name_or_path,
        vae=vae,
        unet=unet,
        controlnet=controlnet, 
    )
    pipe.to(accelerator.device, dtype=weight_dtype)

    image = PIL.Image.open(image_path)

    # Prepare everything with our `accelerator`.
    pipe, image = accelerator.prepare(pipe, image)
    pipe.safety_checker = None

    # Convert image into grayscale
    original_size = image.size
    control_image = image.convert("L").convert("RGB").resize((512, 512))
    
    # Image captioning
    if caption_model_name == "blip-image-captioning-large" or "blip-image-captioning-base":
        caption = blip_image_captioning(control_image, caption_model_name, 
                                        weight_dtype, accelerator.device, conditional=True)
    # elif caption_model_name == "ViT-L-14/openai" or "ViT-H-14/laion2b_s32b_b79k":
    #     caption = clip_image_captioning(control_image, caption_model_name, accelerator.device)
    # elif caption_model_name == "vit-gpt2-image-captioning":
    #     caption = vit_gpt2_image_captioning(control_image, accelerator.device)
    caption = remove_unlikely_words(caption)

    print("================================================================")
    print(f"Positive prompt: \n>>> {positive_prompt}")
    print(f"Negative prompt: \n>>> {negative_prompt}")
    print(f"Caption results: \n>>> {caption}")
    print("================================================================")
    
    # Combine positive prompt and captioning result
    prompt = [positive_prompt + ", " + caption]

    # Image colorization
    image = pipe(prompt=prompt, 
                 negative_prompt=negative_prompt, 
                 num_inference_steps=num_inference_steps, 
                 generator=generator, 
                 image=control_image).images[0]
    
    # Apply color mapping
    result_image = apply_color(control_image, image)
    result_image = result_image.resize(original_size)
    return result_image, caption

def main(args):
    output_image, output_caption = process_image(image_path=args.image_path,
                                                 controlnet_model_name_or_path=args.controlnet_model_name_or_path,
                                                 caption_model_name=args.caption_model_name,
                                                 positive_prompt=args.positive_prompt,
                                                 negative_prompt=args.negative_prompt,
                                                 seed=args.seed,
                                                 num_inference_steps=args.num_inference_steps,
                                                 mixed_precision=args.mixed_precision,
                                                 pretrained_model_name_or_path=args.pretrained_model_name_or_path,
                                                 pretrained_vae_model_name_or_path=args.pretrained_vae_model_name_or_path,
                                                 revision=args.revision,
                                                 variant=args.variant,
                                                 repo=args.repo,
                                                 ckpt=args.ckpt,)
    input_image = PIL.Image.open(args.image_path)
    display_images(input_image.convert("L"), output_image, input_image)
    return output_image, output_caption

# Entry point of the script
if __name__ == "__main__":
    args = parse_args()
    main(args)