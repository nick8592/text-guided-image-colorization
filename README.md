# Text-Guided-Image-Colorization

This project utilizes the power of **Stable Diffusion (SDXL/SDXL-Light)** and the **BLIP (Bootstrapping Language-Image Pre-training)** captioning model to provide an interactive image colorization experience. Users can influence the generated colors of objects within images, making the colorization process more personalized and creative.

## Table of Contents
 - [Features](#features)
 - [Installation](#installation)
 - [Quick Start](#quick-start)
 - [Dataset Usage](#dataset-usage)
 - [Training](#training)
 - [Evaluation](#evaluation)
 - [Results](#results)
 - [License](#license)

## Features

- **Interactive Colorization**: Users can specify desired colors for different objects in the image.
- **ControlNet Approach**: Enhanced colorization capabilities through retraining with ControlNet, allowing SDXL to better adapt to the image colorization task.
- **High-Quality Outputs**: Leverage the latest advancements in diffusion models to generate vibrant and realistic colorizations.
- **User-Friendly Interface**: Easy-to-use interface for seamless interaction with the model.

## Installation

To set up the project locally, follow these steps:

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/nick8592/text-guided-image-colorization.git
   cd text-guided-image-colorization
   ```

2. **Install Dependencies**:
   Make sure you have Python 3.7 or higher installed. Then, install the required packages:

   ```bash
   pip install -r requirements.txt
   ```
   Install `torch` and `torchvision` matching your CUDA version:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cuXXX
   ```
   Replace `XXX` with your CUDA version (e.g., `118` for CUDA 11.8). For more info, see [PyTorch Get Started](https://pytorch.org/get-started/locally/).   


3. **Download Pre-trained Models**:
   | Models | Hugging Face (Recommand) | Other |
   |:---:|:---:|:---:|
   |SDXL-Lightning Caption|[link](https://huggingface.co/nickpai/sdxl_light_caption_output)|[link](https://gofile.me/7uE8s/FlEhfpWPw) (2kNJfV)|
   |SDXL-Lightning Custom Caption (Recommand)|[link](https://huggingface.co/nickpai/sdxl_light_custom_caption_output)|[link](https://gofile.me/7uE8s/AKmRq5sLR) (KW7Fpi)|


   ```bash
   text-guided-image-colorization/sdxl_light_caption_output
   ├── checkpoint-30000
   │   ├── diffusion_pytorch_model.safetensors
   │   └── config.json
   ├── optimizer.bin
   ├── random_states_0.pkl
   ├── scaler.pt
   └── scheduler.bin
   ```

## Quick Start

1. Run the `gradio_ui.py` script:

```bash
python gradio_ui.py
```

2. Open the provided URL in your web browser to access the Gradio-based user interface.

3. Upload an image and use the interface to control the colors of specific objects in the image. But still the model can generate images without a specific prompt.

4. The model will generate a colorized version of the image based on your input (or automatic). See the [demo video](https://x.com/weichenpai/status/1829513077588631987).
![Gradio UI](images/gradio_ui.png)


## Dataset Usage

You can find more details about the dataset usage in the [Dataset-for-Image-Colorization](https://github.com/nick8592/Dataset-for-Image-Colorization).

## Training

For training, you can use one of the following scripts:

- `train_controlnet.sh`: Trains a model using [Stable Diffusion v2](https://huggingface.co/stabilityai/stable-diffusion-2-1)
- `train_controlnet_sdxl.sh`: Trains a model using [SDXL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
- `train_controlnet_sdxl_light.sh`: Trains a model using [SDXL-Lightning](https://huggingface.co/ByteDance/SDXL-Lightning)

Although the training code for SDXL is provided, due to a lack of GPU resources, I wasn't able to train the model by myself. Therefore, there might be some errors when you try to train the model.

## Evaluation

For evaluation, you can use one of the following scripts:

- `eval_controlnet.sh`: Evaluates the model using [Stable Diffusion v2](https://huggingface.co/stabilityai/stable-diffusion-2-1) for a folder of images.
- `eval_controlnet_sdxl_light.sh`: Evaluates the model using [SDXL-Lightning](https://huggingface.co/ByteDance/SDXL-Lightning) for a folder of images.
- `eval_controlnet_sdxl_light_single.sh`: Evaluates the model using [SDXL-Lightning](https://huggingface.co/ByteDance/SDXL-Lightning) for a single image.

## Results
### Prompt-Guided
| Caption | Condition 1 | Condition 2 | Condition 3 |
|:---:|:---:|:---:|:---:|
| ![000000022935_gray.jpg](images/000000022935_gray.jpg) | ![000000022935_green_shirt_on_right_girl.jpeg](images/000000022935_green_shirt_on_right_girl.jpeg) | ![000000022935_purple_shirt_on_right_girl.jpeg](images/000000022935_purple_shirt_on_right_girl.jpeg) |![000000022935_red_shirt_on_right_girl.jpeg](images/000000022935_red_shirt_on_right_girl.jpeg) |
| a photography of a woman in a soccer uniform kicking a soccer ball | + "green shirt"| + "purple shirt" | + "red shirt" |
| ![000000041633_gray.jpg](images/000000041633_gray.jpg) | ![000000041633_bright_red_car.jpeg](images/000000041633_bright_red_car.jpeg) | ![000000041633_dark_blue_car.jpeg](images/000000041633_dark_blue_car.jpeg) |![000000041633_black_car.jpeg](images/000000041633_black_car.jpeg) |
| a photography of a photo of a truck | + "bright red car"| + "dark blue car" | + "black car" |
| ![000000286708_gray.jpg](images/000000286708_gray.jpg) | ![000000286708_orange_hat.jpeg](images/000000286708_orange_hat.jpeg) | ![000000286708_pink_hat.jpeg](images/000000286708_pink_hat.jpeg) |![000000286708_yellow_hat.jpeg](images/000000286708_yellow_hat.jpeg) |
| a photography of a cat wearing a hat on his head | + "orange hat"| + "pink hat" | + "yellow hat" |

### Prompt-Free
Ground truth images are provided solely for reference purpose in the image colorization task.
| Grayscale Image | Colorized Result | Ground Truth |
|:---:|:---:|:---:|
| ![000000025560_gray.jpg](images/000000025560_gray.jpg) | ![000000025560_color.jpg](images/000000025560_color.jpg) | ![000000025560_gt.jpg](images/000000025560_gt.jpg) |
| ![000000065736_gray.jpg](images/000000065736_gray.jpg) | ![000000065736_color.jpg](images/000000065736_color.jpg) | ![000000065736_gt.jpg](images/000000065736_gt.jpg) |
| ![000000091779_gray.jpg](images/000000091779_gray.jpg) | ![000000091779_color.jpg](images/000000091779_color.jpg) | ![000000091779_gt.jpg](images/000000091779_gt.jpg) |
| ![000000092177_gray.jpg](images/000000092177_gray.jpg) | ![000000092177_color.jpg](images/000000092177_color.jpg) | ![000000092177_gt.jpg](images/000000092177_gt.jpg) |
| ![000000166426_gray.jpg](images/000000166426_gray.jpg) | ![000000166426_color.jpg](images/000000166426_color.jpg) | ![000000025560_gt.jpg](images/000000166426_gt.jpg) |



## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
