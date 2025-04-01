import torch
from PIL import Image
from src.pipeline import FluxPipeline
from src.transformer_flux import FluxTransformer2DModel
from src.lora_helper import set_single_lora, set_multi_lora

from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="Xiaojiu-Z/EasyControl", filename="models/canny.safetensors", local_dir="./")
hf_hub_download(repo_id="Xiaojiu-Z/EasyControl", filename="models/inpainting.safetensors", local_dir="./")
hf_hub_download(repo_id="Xiaojiu-Z/EasyControl", filename="models/subject.safetensors", local_dir="./")

def clear_cache(transformer):
    for name, attn_processor in transformer.attn_processors.items():
        attn_processor.bank_kv.clear()

# Initialize model
device = "cuda"
base_path = "black-forest-labs/FLUX.1-dev"  # Path to your base model
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

# Single spatial condition control example
path = control_models["canny"]
set_single_lora(pipe.transformer, path, lora_weights=[1], cond_size=512)
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

image.save("output_multi.png")

# Clear cache after generation
clear_cache(pipe.transformer)

