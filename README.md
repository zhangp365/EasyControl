# Implementation of EasyControl

EasyControl: Adding Efficient and Flexible Control for Diffusion Transformer

<a href='https://easycontrolproj.github.io/'><img src='https://img.shields.io/badge/Project-Page-green'></a>
<a href='https://arxiv.org/pdf/2503.07027'><img src='https://img.shields.io/badge/Technique-Report-red'></a> 
<a href="https://huggingface.co/Xiaojiu-Z/EasyControl/"><img src="https://img.shields.io/badge/🤗_HuggingFace-Model-ffbd45.svg" alt="HuggingFace"></a>
<a href='https://huggingface.co/spaces/jamesliu1217/EasyControl'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue'></a>

> *[Yuxuan Zhang](https://xiaojiu-z.github.io/YuxuanZhang.github.io/), [Yirui Yuan](https://github.com/Reynoldyy), [Yiren Song](https://scholar.google.com.hk/citations?user=L2YS0jgAAAAJ), [Haofan Wang](https://haofanwang.github.io/), [Jiaming Liu](https://scholar.google.com/citations?user=SmL7oMQAAAAJ&hl=en)*
> <br>
> Tiamat AI, ShanghaiTech University, National University of Singapore, Liblib AI

<img src='assets/teaser.jpg'>

## Features
* **Motivation:** The architecture of diffusion models is transitioning from Unet-based to DiT (Diffusion Transformer). However, the DiT ecosystem lacks mature plugin support and faces challenges such as efficiency bottlenecks, conflicts in multi-condition coordination, and insufficient model adaptability.
* **Contribution:** We propose EasyControl, an efficient and flexible unified conditional DiT framework. By incorporating a lightweight Condition Injection LoRA module, a Position-Aware Training Paradigm, and a combination of Causal Attention mechanisms with KV Cache technology, we significantly enhance **model compatibility** (enabling plug-and-play functionality and style lossless control), **generation flexibility** (supporting multiple resolutions, aspect ratios, and multi-condition combinations), and **inference efficiency**.
<img src='assets/method.jpg'>

## News
- **2025-03-12**: ⭐️ Inference code are released. Once we have ensured that everything is functioning correctly, the new model will be merged into this repository. Stay tuned for updates! 😊
- **2025-03-18**: 🔥 We have released our [pre-trained checkpoints](https://huggingface.co/Xiaojiu-Z/EasyControl/) on Hugging Face! You can now try out EasyControl with the official weights.
- **2025-03-19**: 🔥 We have released [huggingface demo](https://huggingface.co/spaces/jamesliu1217/EasyControl)! You can now try out EasyControl with the huggingface space, enjoy it!
<div align="center">
<table>
<tr>
    <td><img src="assets/example1.jpeg" alt="Example 1" width="400"/></td>
    <td><img src="assets/example2.jpeg" alt="Example 2" width="400"/></td>
</tr>
<tr>
    <td align="center">Example 1</td>
    <td align="center">Example 2</td>
</tr>
</table>
</div>

## Installation

We recommend using Python 3.10 and PyTorch with CUDA support. To set up the environment:

```bash
# Create a new conda environment
conda create -n easycontrol python=3.10
conda activate easycontrol

# Install other dependencies
pip install -r requirements.txt
```

## Download

You can download the model directly from [Hugging Face](https://huggingface.co/EasyControl/EasyControl).
Or download using Python script:

```python
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="Xiaojiu-Z/EasyControl", filename="models/canny.safetensors", local_dir="./")
hf_hub_download(repo_id="Xiaojiu-Z/EasyControl", filename="models/depth.safetensors", local_dir="./")
hf_hub_download(repo_id="Xiaojiu-Z/EasyControl", filename="models/hedsketch.safetensors", local_dir="./")
hf_hub_download(repo_id="Xiaojiu-Z/EasyControl", filename="models/inpainting.safetensors", local_dir="./")
hf_hub_download(repo_id="Xiaojiu-Z/EasyControl", filename="models/pose.safetensors", local_dir="./")
hf_hub_download(repo_id="Xiaojiu-Z/EasyControl", filename="models/seg.safetensors", local_dir="./")
hf_hub_download(repo_id="Xiaojiu-Z/EasyControl", filename="models/subject.safetensors", local_dir="./")
```

If you cannot access Hugging Face, you can use [hf-mirror](https://hf-mirror.com/) to download the models:
```python
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download --resume-download Xiaojiu-Z/EasyControl --local-dir checkpoints --local-dir-use-symlinks False
```

## Usage
Here's a basic example of using EasyControl:

### Model Initialization

```python
import torch
from PIL import Image
from src.pipeline import FluxPipeline
from src.transformer_flux import FluxTransformer2DModel
from src.lora_helper import set_single_lora, set_multi_lora

def clear_cache(transformer):
    for name, attn_processor in transformer.attn_processors.items():
        attn_processor.bank_kv.clear()

# Initialize model
device = "cuda"
base_path = "FLUX.1-dev"  # Path to your base model
pipe = FluxPipeline.from_pretrained(base_path, torch_dtype=torch.bfloat16, device=device)
transformer = FluxTransformer2DModel.from_pretrained(
    base_path, 
    subfolder="transformer",
    torch_dtype=torch.bfloat16, 
    device=device
)
pipe.transformer = transformer
pipe.to(device)

# Load control models
lora_path = "./models"
control_models = {
    "canny": f"{lora_path}/canny.safetensors",
    "depth": f"{lora_path}/depth.safetensors",
    "hedsketch": f"{lora_path}/hedsketch.safetensors",
    "pose": f"{lora_path}/pose.safetensors",
    "seg": f"{lora_path}/seg.safetensors",
    "inpainting": f"{lora_path}/inpainting.safetensors",
    "subject": f"{lora_path}/subject.safetensors",
}
```

### Single Condition Control

```python
# Single spatial condition control example
path = control_models["canny"]
set_single_lora(pipe.transformer, path, lora_weights=[1], cond_size=512)

# Generate image
prompt = "A nice car on the beach"
spatial_image = Image.open("./test_imgs/canny.png").convert("RGB")

image = pipe(
    prompt,
    height=720,
    width=992,
    guidance_scale=3.5,
    num_inference_steps=25,
    max_sequence_length=512,
    generator=torch.Generator("cpu").manual_seed(5),
    spatial_images=[spatial_image],
    cond_size=512,
).images[0]

# Clear cache after generation
clear_cache(pipe.transformer)
```

<div align="center">
<table>
<tr>
    <td><img src="test_imgs/canny.png" alt="Canny Condition" width="400"/></td>
    <td><img src="assets/result_canny.png" alt="Generated Result" width="400"/></td>
</tr>
<tr>
    <td align="center">Canny Condition</td>
    <td align="center">Generated Result</td>
</tr>
</table>
</div>

```python
# Single subject condition control example
path = control_models["subject"]
set_single_lora(pipe.transformer, path, lora_weights=[1], cond_size=512)

# Generate image
prompt = "A SKS in the library"
subject_image = Image.open("./test_imgs/subject_0.png").convert("RGB")

image = pipe(
    prompt,
    height=1024,
    width=1024,
    guidance_scale=3.5,
    num_inference_steps=25,
    max_sequence_length=512,
    generator=torch.Generator("cpu").manual_seed(5),
    subject_images=[subject_image],
    cond_size=512,
).images[0]

# Clear cache after generation
clear_cache(pipe.transformer)
```

<div align="center">
<table>
<tr>
    <td><img src="test_imgs/subject_0.png" alt="Subject Condition" width="400"/></td>
    <td><img src="assets/result_subject.png" alt="Generated Result" width="400"/></td>
</tr>
<tr>
    <td align="center">Subject Condition</td>
    <td align="center">Generated Result</td>
</tr>
</table>
</div>

### Multi-Condition Control

```python
# Multi-condition control example
paths = [control_models["subject"], control_models["inpainting"]]
set_multi_lora(pipe.transformer, paths, lora_weights=[[1], [1]], cond_size=512)

prompt = "A SKS on the car"
subject_images = [Image.open("./test_imgs/subject_1.png").convert("RGB")]
spatial_images = [Image.open("./test_imgs/inpainting.png").convert("RGB")]

image = pipe(
    prompt,
    height=1024,
    width=1024,
    guidance_scale=3.5,
    num_inference_steps=25,
    max_sequence_length=512,
    generator=torch.Generator("cpu").manual_seed(42),
    subject_images=subject_images,
    spatial_images=spatial_images,
    cond_size=512,
).images[0]

# Clear cache after generation
clear_cache(pipe.transformer)
```

<div align="center">
<table>
<tr>
    <td><img src="test_imgs/subject_1.png" alt="Subject Condition" width="250"/></td>
    <td><img src="test_imgs/inpainting.png" alt="Inpainting Condition" width="250"/></td>
    <td><img src="assets/result_subject_inpainting.png" alt="Generated Result" width="250"/></td>
</tr>
<tr>
    <td align="center">Subject Condition</td>
    <td align="center">Inpainting Condition</td>
    <td align="center">Generated Result</td>
</tr>
</table>
</div>


## Usage Tips

- Clear cache after each generation using `clear_cache(pipe.transformer)`
- For optimal performance:
  - Start with `guidance_scale=3.5` and adjust based on results
  - Use `num_inference_steps=25` for a good balance of quality and speed
- When using set_multi_lora api, make sure the subject lora path(subject) is before the spatial lora path(canny, depth, hedsketch, etc.).

## Todo List
1. - [x] Inference code 
2. - [x] Spatial Pre-trained weights 
3. - [x] Subject Pre-trained weights 
4. - [ ] Training code


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Xiaojiu-z/EasyControl&type=Date)](https://star-history.com/#Xiaojiu-z/EasyControl&Date)

## Disclaimer
The code of EasyControl is released under [Apache License](https://github.com/Xiaojiu-Z/EasyControl?tab=Apache-2.0-1-ov-file#readme) for both academic and commercial usage. Our released checkpoints are for research purposes only. Users are granted the freedom to create images using this tool, but they are obligated to comply with local laws and utilize it responsibly. The developers will not assume any responsibility for potential misuse by users.


## Citation
```
@misc{zhang2025easycontroladdingefficientflexible,
      title={EasyControl: Adding Efficient and Flexible Control for Diffusion Transformer}, 
      author={Yuxuan Zhang and Yirui Yuan and Yiren Song and Haofan Wang and Jiaming Liu},
      year={2025},
      eprint={2503.07027},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2503.07027}, 
}
```
