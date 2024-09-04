import PIL
import torch
import subprocess
import gradio as gr

from typing import Optional
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
from clip_interrogator import Interrogator, Config, list_clip_models

def apply_color(image: PIL.Image.Image, color_map: PIL.Image.Image) -> PIL.Image.Image:
    # Convert input images to LAB color space
    image_lab = image.convert('LAB')
    color_map_lab = color_map.convert('LAB')

    # Split LAB channels
    l, a , b = image_lab.split()
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

# def vit_gpt2_image_captioning(image: PIL.Image.Image, device: str) -> str:
#     # https://huggingface.co/nlpconnect/vit-gpt2-image-captioning
#     model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning").to(device)
#     feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
#     tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

#     max_length = 16
#     num_beams = 4
#     gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

#     pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values
#     pixel_values = pixel_values.to(device)

#     output_ids = model.generate(pixel_values, **gen_kwargs)

#     preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
#     caption = [pred.strip() for pred in preds]

#     return caption[0]

# def clip_image_captioning(image: PIL.Image.Image,
#                           clip_model_name: str,
#                           device: str) -> str:
#     # validate clip model name
#     models = list_clip_models()
#     if clip_model_name not in models:
#         raise ValueError(f"Could not find CLIP model {clip_model_name}! \
#                          Available models: {models}")
#     config = Config(device=device, clip_model_name=clip_model_name)
#     config.apply_low_vram_defaults()
#     ci = Interrogator(config)
#     caption = ci.interrogate(image)
#     return caption

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

# Define the image gallery based on folder path
def get_image_paths(folder_path):
  import os
  image_paths = []
  for filename in os.listdir(folder_path):
    if filename.endswith(".jpg") or filename.endswith(".png"):
      image_paths.append([os.path.join(folder_path, filename)])
  return image_paths

# Create the Gradio interface
def create_interface():
    controlnet_model_dict = {
       "sdxl-light-caption-30000": "sdxl_light_caption_output/checkpoint-30000/controlnet",
       "sdxl-light-custom-caption-30000": "sdxl_light_custom_caption_output/checkpoint-30000/controlnet",
    }
    images = get_image_paths("example/legacy_images")  # Replace with your folder path

    interface = gr.Interface(
        fn=process_image,
        inputs=[
            gr.Image(label="Upload image", 
                     value="example/legacy_images/Hollywood-Sign.jpg",
                     type='filepath'),
            gr.Dropdown(choices=[controlnet_model_dict[key] for key in controlnet_model_dict], 
                        value=controlnet_model_dict["sdxl-light-caption-30000"],
                        label="Select ControlNet Model"),
            gr.Dropdown(choices=["blip-image-captioning-large",
                                 "blip-image-captioning-base",], 
                        value="blip-image-captioning-large",
                        label="Select Image Captioning Model"),
            gr.Textbox(label="Positive Prompt", placeholder="Text for positive prompt"),
            gr.Textbox(value="low quality, bad quality, low contrast, black and white, bw, monochrome, grainy, blurry, historical, restored, desaturate",
                       label="Negative Prompt", placeholder="Text for negative prompt"),
        ],
        outputs=[
            gr.Image(label="Colorized image", 
                     value="example/UUColor_results/Hollywood-Sign.jpeg",
                     format="jpeg"),
            gr.Textbox(label="Captioning Result", show_copy_button=True)
        ],
        examples=images,
        additional_inputs=[
            # gr.Radio(choices=["Original", "Square"], value="Original", 
            #          label="Output resolution"),
            # gr.Slider(minimum=128, maximum=512, value=256, step=128,
            #           label="Height & Width", 
            #           info='Only effect if select "Square" output resolution'),
            gr.Slider(0, 1000, 123, label="Seed"),
            gr.Radio(choices=[1, 2, 4, 8],
                     value=8,
                     label="Inference Steps",
                     info="1-step, 2-step, 4-step, or 8-step distilled models"),
            gr.Radio(choices=["no", "fp16", "bf16"],
                     value="fp16",
                     label="Mixed Precision",
                     info="Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16)."),
            gr.Dropdown(choices=["stabilityai/stable-diffusion-xl-base-1.0"],
                        value="stabilityai/stable-diffusion-xl-base-1.0",
                        label="Base Model",
                        info="Path to pretrained model or model identifier from huggingface.co/models."),
            gr.Dropdown(choices=["None"],
                        value=None,
                        label="VAE Model",
                        info="Path to an improved VAE to stabilize training. For more details check out: https://github.com/huggingface/diffusers/pull/4038."),
            gr.Dropdown(choices=["None"],
                        value=None,
                        label="Varient",
                        info="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16"),
            gr.Dropdown(choices=["None"],
                        value=None,
                        label="Revision",
                        info="Revision of pretrained model identifier from huggingface.co/models."),
            gr.Dropdown(choices=["ByteDance/SDXL-Lightning"],
                        value="ByteDance/SDXL-Lightning",
                        label="Repository",
                        info="Repository from huggingface.co"),
            gr.Dropdown(choices=["sdxl_lightning_1step_unet.safetensors",
                                 "sdxl_lightning_2step_unet.safetensors",
                                 "sdxl_lightning_4step_unet.safetensors",
                                 "sdxl_lightning_8step_unet.safetensors"],
                        value="sdxl_lightning_8step_unet.safetensors",
                        label="Checkpoint",
                        info="Available checkpoints from the repository. Caution! Checkpoint's 'N'step must match with inference steps"),
        ],
        title="Text-Guided Image Colorization",
        description="Upload an image and select a model to colorize it."
    )
    return interface

def main():
    # Launch the Gradio interface
    interface = create_interface()
    interface.launch()

if __name__ == "__main__":
   main()
